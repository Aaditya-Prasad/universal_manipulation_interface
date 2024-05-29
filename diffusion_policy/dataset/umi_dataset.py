import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
import h5py
import multiprocessing
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
import concurrent.futures


from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()


def convert_robomimic_to_replay(store, shape_meta, dataset_path, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            if key.endswith('wrt_start'):
                continue
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                # this_data = _convert_actions(
                #     raw_actions=this_data,
                #     abs_action=abs_action,
                #     rotation_transformer=rotation_transformer
                # )
                # assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])

                # We now handle conversion within get item!
                # We can't assert shape here because rot will still be in axis angle, conversion to 
                # 6d happens in get item
                pass
            else:
                if not (this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape']) \
                        or (this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['raw_shape']))): # not all have raw shape, intended short-circuit
                    raise RuntimeError(f"shape mismatch for {key}: {this_data.shape} vs {(n_steps,) + tuple(shape_meta['obs'][key]['shape'])}")
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

class UmiDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'relative')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'relative')
        
        # if cache_dir is None:
        #     # load into memory store
        #     with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        #         replay_buffer = ReplayBuffer.copy_from_store(
        #             src_store=zip_store, 
        #             store=zarr.MemoryStore()
        #         )
        # else:
        #     # TODO: refactor into a stand alone function?
        #     # determine path name
        #     mod_time = os.path.getmtime(dataset_path)
        #     stamp = datetime.fromtimestamp(mod_time).isoformat()
        #     stem_name = os.path.basename(dataset_path).split('.')[0]
        #     cache_name = '_'.join([stem_name, stamp])
        #     cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
        #     cache_dir.mkdir(parents=True, exist_ok=True)
        #     cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
        #     lock_path = cache_dir.joinpath(cache_name + '.lock')
            
        #     # load cached file
        #     print('Acquiring lock on cache.')
        #     with FileLock(lock_path):
        #         # cache does not exist
        #         if not cache_path.exists():
        #             try:
        #                 with zarr.LMDBStore(str(cache_path),     
        #                     writemap=True, metasync=False, sync=False, map_async=True, lock=False
        #                     ) as lmdb_store:
        #                     with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        #                         print(f"Copying data to {str(cache_path)}")
        #                         ReplayBuffer.copy_from_store(
        #                             src_store=zip_store,
        #                             store=lmdb_store
        #                         )
        #                 print("Cache written to disk!")
        #             except Exception as e:
        #                 shutil.rmtree(cache_path)
        #                 raise e
            
        #     # open read-only lmdb store
        #     store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
        #     replay_buffer = ReplayBuffer.create_from_group(
        #         group=zarr.group(store)
        #     )
        replay_buffer = None
        cache_zarr_path = dataset_path + '.zarr.zip'
        cache_lock_path = cache_zarr_path + '.lock'
        print('Acquiring lock on cache.')
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                # cache does not exists
                try:
                    print('Cache does not exist. Creating!')
                    # store = zarr.DirectoryStore(cache_zarr_path)
                    replay_buffer = convert_robomimic_to_replay(
                        store=zarr.MemoryStore(), 
                        shape_meta=dataset_shape_meta, 
                        dataset_path=dataset_path,
                        )
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(
                            store=zip_store
                        )
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print('Loading cached ReplayBuffer from Disk.')
                with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore())
                print('Loaded!')

        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)


            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in dataset_shape_meta['obs'].keys():
            print(f"KEY: {key}")
            if key.endswith('_demo_start_pos') or key.endswith('_demo_end_pos'):
                self.sampler_lowdim_keys.append(key)
                print(f"AUTO-ADDED {key} to lowdim_keys")
                query_key = 'arm_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    

    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                if(key in batch['obs'].keys()):
                    data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        action_normalizers = list()
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][...,:6])))              # pos
        action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., 6:12]))) # rot
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., 12:])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot') or 'rot_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pose'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # generate relative pose with respect to episode start
        # HACK: add noise to episode start pose
        # if (f'robot{other_robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
        #     (f'robot{other_robot_id}_eef_rot_wrt_start' not in self.shape_meta['obs']):
        #     continue
        
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            obs_dict[f'arm_pos'],
            obs_dict[f'arm_rot']
        ], axis=-1))
        
        # get start pose
        start_pose = obs_dict[f'arm_demo_start_pos'][0]
        # HACK: add noise to episode start pose
        start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
        start_pose_mat = pose_to_mat(start_pose)
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            ref_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)
        
        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
        obs_dict[f'arm_rot_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pos') or key.endswith('_demo_end_pos'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

            # convert pose to mat
        arm_pose_mat = pose_to_mat(np.concatenate([
            obs_dict[f'arm_pos'],
            obs_dict[f'arm_rot']
        ], axis=-1))
        base_pose_obs = obs_dict['base_pose']

        arm_action = data['action'][..., 3:9]
        arm_action_mat = pose_to_mat(arm_action)   
        base_action = data['action'][..., :3]

            
        # solve relative obs
        arm_obs_pose_mat = convert_pose_mat_rep(
            arm_pose_mat, 
            ref_pose_mat=arm_pose_mat[-1],
            pose_rep=self.action_pose_repr,
            backward=False)
        arm_action_pose_mat = convert_pose_mat_rep(
            arm_action_mat, 
            ref_pose_mat=arm_pose_mat[-1],
            pose_rep=self.obs_pose_repr,
            backward=False)
        
        base_action_rel = base_action - base_pose_obs[-1]
        base_obs_rel = base_pose_obs - base_pose_obs[-1]
        
        # convert pose to pos + rot6d representation
        obs_pose = mat_to_pose10d(arm_obs_pose_mat)
        arm_action_pose = mat_to_pose10d(arm_action_pose_mat)
        
        action_gripper = data['action'][..., -1:]
        data['action'] = (np.concatenate([base_action_rel, arm_action_pose, action_gripper], axis=-1))

        # generate data
        obs_dict[f'arm_pos'] = obs_pose[:,:3]
        obs_dict[f'arm_rot'] = obs_pose[:,3:]
        obs_dict[f'base_pose'] = base_obs_rel
            
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
