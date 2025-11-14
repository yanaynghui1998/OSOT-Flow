# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

from .util import unsqueeze_xdim
from dataset.bratsloader2 import TorchLowResolution as LR_opt
from ipdb import set_trace as debug
def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var,sigma1

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var, sigma1 = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)
        self.sigma1=to_torch(sigma1).to(device)
        # Low-resolution opeartaion
        self.LR=LR_opt(factor=4)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)
        beta=unsqueeze_xdim(self.betas[step], xdim)
        xt = mu_x0 * x0 + mu_x1 * x1#flow
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach(), mu_x1, beta

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def split_volume_into_patches_torch(self, volume, patch_size):

        _, _, D, H, W = volume.shape 
        assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0

        patches = []
        
        for d in range(0, D, patch_size):
            for h in range(0, H, patch_size):
                for w in range(0, W, patch_size):
                    patch = volume[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patches.append(patch)
        
        return torch.stack(patches, dim=0)


    def reconstruct_volume_from_patches_torch(self, patches, original_shape, patch_size):

        D, H, W = original_shape
        assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0

        # 计算 Patch 数量
        num_d = D // patch_size
        num_h = H // patch_size
        num_w = W // patch_size

        # 初始化空的体积数据
        volume = torch.zeros((1, 1, D, H, W)).to(self.device)

        index = 0  # Patch 索引
        for d in range(num_d):
            for h in range(num_h):
                for w in range(num_w):
                    volume[
                        :, :, 
                        d * patch_size: (d + 1) * patch_size,
                        h * patch_size: (h + 1) * patch_size,
                        w * patch_size: (w + 1) * patch_size
                    ] = patches[index]
                    index += 1

        return volume

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    def ddpm_dps_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True,step_size=1.0):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            xt=xt.requires_grad_(True)

            xt_patches=self.split_volume_into_patches_torch(xt, patch_size=16)
            xt_list=[]
            pred_x0_list=[]

            for _,xt_patch in enumerate(xt_patches):

                pred_x0_patch = pred_x0_fn(xt_patch, step)
                pred_x0_list.append(pred_x0_patch)

            pred_x0=self.reconstruct_volume_from_patches_torch(pred_x0_list, original_shape=(64,64,64), patch_size=16)

            # corrupt_x0_forw=self.LR(pred_x0)
            
            # residual = corrupt_x0_forw - x1
            # residual_norm = (torch.linalg.norm(residual) ** 2).sum()

            # norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xt)[0]
            
            pred_x0_patch=self.split_volume_into_patches_torch(pred_x0, patch_size=16)

            for xt_patch, pred_x0_patch in zip(xt_patches,pred_x0_patch):
                xt_patch_pre = self.p_posterior(prev_step, step, xt_patch, pred_x0_patch, ot_ode=ot_ode)
                xt_list.append(xt_patch_pre)

            xt=self.reconstruct_volume_from_patches_torch(xt_list, original_shape=(64,64,64), patch_size=16)
            # xt = xt - step_size * norm_grad

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    def ddpm_dps_patch_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True,step_size=1.0):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            # xt=xt.requires_grad_()
            xt=xt.requires_grad_(True)
            pred_x0 = pred_x0_fn(xt, step)

            corrupt_x0_forw=self.LR(pred_x0)
            
            residual = corrupt_x0_forw - x1
            residual_norm = (torch.linalg.norm(residual) ** 2).sum()

            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xt)[0]

            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            xt = xt - step_size * norm_grad

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)