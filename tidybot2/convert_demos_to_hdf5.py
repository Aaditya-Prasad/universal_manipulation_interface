# Adapted from:
# - https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py
# - https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/split_train_val.py

import argparse
import pickle
from pathlib import Path
import cv2 as cv
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from constants import POLICY_IMAGE_WIDTH

def read_frames_from_mp4(mp4_path):
    cap = cv.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB))
    cap.release()
    return frames

def gather_demos_as_hdf5(input_dir, hdf5_path):
    # Get list of episode dirs
    episode_dirs = sorted([child for child in Path(input_dir).iterdir() if child.is_dir()])

    # Convert to HDF5 format
    with h5py.File(hdf5_path, 'w') as f:
        data_group = f.create_group('data')

        # Iterate through episodes
        for episode_idx, episode_dir in enumerate(tqdm(episode_dirs)):
            # Load data
            with open(episode_dir / 'data.pkl', 'rb') as f:
                data = pickle.load(f)

            # Check whether episode is empty
            if len(data['observations']) == 0:
                print(f'Warning: Skipping {episode_dir} as it contains no data')
                continue

            # Extract observations
            observations = {}
            frames_dict = {}
            for step_idx, obs in enumerate(data['observations']):
                for k, v in obs.items():
                    # Images are stored as MP4 videos
                    if v is None:
                        # Load images from MP4 file
                        if k not in frames_dict:
                            mp4_path = episode_dir / f'{k}.mp4'
                            frames_dict[k] = read_frames_from_mp4(mp4_path)

                        # Get image for current step
                        v = frames_dict[k][step_idx]

                        # Resize image
                        v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_WIDTH))

                    # Append extracted observation
                    if k == 'arm_quat':
                        k = 'arm_rot'
                        v = Rotation.from_quat(v).as_rotvec()

                    if k not in observations:
                        observations[k] = []
                    observations[k].append(v)

            k = 'arm_demo_start_pos'
            v = np.concatenate((
                observations['arm_pos'][0],
                observations['arm_rot'][0],
            ), axis=0)
            observations[k] = [v] * len(observations['arm_pos'])

            # Extract actions
            actions = [
                np.concatenate((
                    action['base_pose'],
                    action['arm_pos'],
                    Rotation.from_quat(action['arm_quat']).as_rotvec(),
                    action['gripper_pos'],
                )) for action in data['actions']
            ]

            # Write to HDF5
            episode_key = f'demo_{episode_idx}'
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f'obs/{k}', data=np.array(v))
            episode_group.create_dataset('actions', data=np.array(actions))

def store_hdf5_filter_keys(hdf5_path, demo_keys, key_name):
    with h5py.File(hdf5_path, 'a') as f:
        # Store filter keys under mask group
        mask_key = f'mask/{key_name}'
        f[mask_key] = np.array(demo_keys, dtype='S')

def split_train_val_from_hdf5(hdf5_path, val_ratio=0):
    with h5py.File(hdf5_path, 'r') as f:
        demos = sorted(list(f['data'].keys()))
    num_demos = len(demos)

    # Need to have at least 1 train and 1 val
    assert num_demos >= 2
    num_val = int(val_ratio * num_demos)
    num_val = max(1, num_val)

    # Get random split
    mask = np.zeros(num_demos, dtype=np.int32)
    mask[:num_val] = 1
    rng = np.random.default_rng(seed=0)
    rng.shuffle(mask)
    train_inds = (1 - mask).nonzero()[0]
    val_inds = mask.nonzero()[0]
    train_keys = [demos[i] for i in train_inds]
    val_keys = [demos[i] for i in val_inds]
    print(f'{num_val} validation demos out of {num_demos} total demos.')

    # Store filter keys
    store_hdf5_filter_keys(hdf5_path=hdf5_path, demo_keys=train_keys, key_name='train')
    store_hdf5_filter_keys(hdf5_path=hdf5_path, demo_keys=val_keys, key_name='valid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='data/demos')
    parser.add_argument('--output-path', default='data/demos.hdf5')
    args = parser.parse_args()
    gather_demos_as_hdf5(args.input_path, args.output_path)
    split_train_val_from_hdf5(args.output_path)
