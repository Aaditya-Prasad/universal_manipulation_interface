On the left is the reference file in mb/ which come largely from Consistency Policy, on the right is the UMI file that was changed directly 
- base_workspace -> workspace/base_workspace, changed to take training.output_dir
- mb_diffusion_hybrid_workspace -> train_diffusion_unet_image_workspace, the latter includes new code from UMI for the policy we're using, we add inference mode, eval mode, etc. 
- 6dpos_dataset -> umi_dataset.py, the latter includes fixed relative obs + actions

- N/A -> diffusion_unet_timm_policy.py, this is new from UMI, has a new obs encoder and perturbs the diffusion pos. Not directly changed but is important. Note that we don't currently use it in our workspace, we use diffusion_unet_image_policy.py instead
- N/A -> tidybot2.yaml, new config file in diffusion_policy/config/task/
- N/A -> train_uner_tidybot2.yaml, new config file in diffusion_policy/config

- utils -> N/A, utils is new and we leave it in this folder
- policy wrapper -> N/A, policy wrapper is new and we leave it in this folder
- example.ipynb -> N/A, ...

Note that we have 
- mb_diffusion_hybrid_workspace which uses the original robomimic obs encoder (don't use this)
- train_unet_tidybot2.yaml, which uses the multi image obs encoder, which works with general resolution
- train_unet_timm_tidybot2.yaml, which uses the timm obs encoder, which works with 224x224 images but is probably more powerful

### Keys:
- base_pose [T, 3]
- arm_pos [T, 3]
- arm_rot_axis_angle [T, 3]
- gripper_pos [T, 1]
- arm_demo_start_pos [T, 6] (3 dims for arm_pos, 3 dims for arm_rot_axis_angle, also T can just be the same 6 values copied T times)
- action [T, 10] (3 base, 3 arm, 3 aa arm, 1 gripper): will be converted to [T, 13] for policy fitting since aa -> 6d

input to the policy has 19 low_dim shape and 2 images, see example.ipynb

### TODOs:
- Val loss is commented out in the workspace code, unsure why

