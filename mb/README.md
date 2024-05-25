On the left is the reference file in mb/ which come largely from Consistency Policy, on the right is the UMI file that was changed directly 
base_workspace -> workspace/base_workspace, changed to take training.output_dir
mb_diffusion_hybrid_workspace -> train_diffusion_unet_image_workspace, the latter includes new code from UMI for the policy we're using, we add inference mode, eval mode, etc. 
6dpos_dataset -> umi_dataset.py, the latter includes fixed relative obs + actions

N/A -> diffusion_unet_timm_policy.py, this is new from UMI, has a new obs encoder and perturbs the diffusion pos. Not directly changed but is important. 

utils -> N/A, utils is new and we leave it in this folder
policy wrapper -> N/A, policy wrapper is new and we leave it in this folder
example.ipynb -> N/A, ...

TODOs:
- Val loss is commented out in the workspace code, unsure why