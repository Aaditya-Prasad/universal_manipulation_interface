import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import IPython
e = IPython.embed
from .models import build_ACT_model, build_CNNMLP_model
from typing import Dict
from diffusion_policy.model.common.normalizer import LinearNormalizer, DummyNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from omegaconf import OmegaConf
from collections import OrderedDict

class ACTPolicy(BaseImagePolicy):
    # this should be the main args, with both args.policy and args.optimizer
    def __init__(self, args: OmegaConf):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args.training.kl_weight
        self.vq = args.policy.vq
        self.normalizer = LinearNormalizer()
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, obs, actions=None, is_pad=None, vq_sample=None):

        env_state = None
        nobs = self.normalizer(obs)

        low_dim_obs = torch.cat((nobs['base_pose'], nobs['arm_pos'], nobs['arm_rot'], nobs['arm_rot_wrt_start'], nobs['gripper_pos']), dim=-1) # b 19
        if len(low_dim_obs.shape) == 3:
            low_dim_obs = low_dim_obs.squeeze(1) # actpp only takes obs history of 1, but the policy wrapper doesn't know this  
        images = torch.cat((nobs['base_image'], nobs['wrist_image']), dim=1) # b 2 c h w
        
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(low_dim_obs, images, env_state, actions, is_pad, vq_sample)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _, _ = self.model(low_dim_obs, images, env_state, vq_sample=vq_sample) # no action, sample from prior
            return a_hat


    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, low_dim_obs, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(low_dim_obs, actions, is_pad)

        return binaries
        
    def serialize(self):    
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
    
    # DIFFUSION POLICY API ########################################################
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix=None) -> Dict[str, torch.Tensor]: # fixed_action_prefix only for API compatibility
        action = self(obs_dict, actions=None)
        return {"action": action, "action_pred": action} # means action and prediction horizon must be equal
        
    def compute_loss(self, batch):
        obs = batch['obs']
        nactions = self.normalizer(batch['actions'])
        return self(obs, nactions)
    
    def set_normalizer(self, normalizer: LinearNormalizer, dummy=False):
        if dummy:
            self.normalizer = DummyNormalizer()
        else:
            self.normalizer.load_state_dict(normalizer.state_dict())


def build_ACT_model_and_optimizer(args: OmegaConf):

    opt_args = args.optimizer
    policy_args = args.policy

    model = build_ACT_model(policy_args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": opt_args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt_args.lr,
                                  weight_decay=opt_args.weight_decay)

    return model, optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
