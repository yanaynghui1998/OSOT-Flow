# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
# from evaluation import build_resnet50
from dataset.bratsloader2 import TorchLowResolution as LR
from . import util
from .network import Image256Net
from .diffusion import Diffusion
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .contrastive_util import normalized_feature_distance,normalized_feature_distance_l1,ContrastLoss,HCRLoss3D,HCRLoss3D_Cosine,ContrastLoss_Grad,HCRLoss3D_Triplet
from ptflops import get_model_complexity_info
# from ipdb import set_trace as debug
def safe_load_optimizer(optimizer, checkpoint, log):
    if "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info("[Opt] Optimizer state loaded successfully.")

            # 修复学习率覆盖的问题（可选）
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group.get('lr', 1e-4)

        except ValueError as e:
            log.warning(f"[Opt] Skipped loading optimizer: {e}")
    else:
        log.warning("[Opt] No optimizer state in checkpoint.")

    return optimizer

def build_optimizer_sched(opt, net, log, extra_params=None):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}

    if extra_params is not None:
        params = list(net.parameters()) + list(extra_params)
    else:
        params = net.parameters()

    optimizer = AdamW(params, **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        optimizer = safe_load_optimizer(optimizer, checkpoint, log)

        if sched is not None and "sched" in checkpoint and checkpoint["sched"] is not None:
            try:
                sched.load_state_dict(checkpoint["sched"])
                log.info("[Opt] LR scheduler loaded from checkpoint.")
            except Exception as e:
                log.warning(f"[Opt] Failed to load scheduler: {e}")
        else:
            log.warning(f"[Opt] No scheduler in checkpoint.")

    return optimizer, sched


def build_fake_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr_fake, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    # log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        # log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            for param_group in optimizer.param_groups:
                print(f"Resetting lr: {param_group['lr']} → {opt.lr}")
                param_group['lr'] = opt.lr_fake
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])

    return optimizer, sched


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

import torch

def manual_mse_loss_pure(input, target, reduction='mean'):
    """
    手动计算 MSE 损失，连 mean/sum 都手动实现。

    参数:
        input (Tensor): 模型输出，与 target 形状相同
        target (Tensor): 标签数据
        reduction (str): 'mean', 'sum', or 'none'

    返回:
        Tensor or float
    """
    # 计算平方误差
    diff = input - target
    squared_error = diff * diff  # 手动平方

    if reduction == 'none':
        return squared_error

    # 获取总元素数
    num_elements = 1
    for dim in squared_error.shape:
        num_elements *= dim
    print(num_elements)
    # 手动 sum
    total_error = torch.zeros(1, dtype=input.dtype, device=input.device)
    for val in squared_error.view(-1):
        total_error += val
    print(total_error)
    if reduction == 'sum':
        return total_error
    elif reduction == 'mean':
        return total_error / num_elements
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

        '''
        student model
        '''
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        '''
        teacher model real
        '''
        self.net_real = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema_real = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        '''
        teacher model fake
        '''
        self.net_fake = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema_fake = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)


        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

            self.net_real.load_state_dict(checkpoint['net'])
            log.info(f"[RealNet] Loaded network ckpt: {opt.load}!")
            self.ema_real.load_state_dict(checkpoint["ema"])
            log.info(f"[RealEma] Loaded ema ckpt: {opt.load}!")

            self.net_fake.load_state_dict(checkpoint['net'])
            log.info(f"[FakeNet] Loaded network ckpt: {opt.load}!")
            self.ema_fake.load_state_dict(checkpoint["ema"])
            log.info(f"[FakeEma] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.net_real.to(opt.device)
        self.ema_real.to(opt.device)

        self.net_fake.to(opt.device)
        self.ema_fake.to(opt.device)

        self.log = log
        self.LR=LR(factor=3)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    def save_learnable_state(self, learnable_std):
        return {"learnable_std": learnable_std.detach().cpu()}

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0
        return pred_x0
    
    def compute_learnable_pred_x0(self, step, xt, net_out, clip_denoise=False):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])  # [B,1,1,1,1]
        learned_std = std_fwd * self.learnable_std  # 使用 learnable 缩放因子
        pred_x0 = xt - learned_std * net_out

        if clip_denoise:
            pred_x0 = pred_x0.clamp(0, 1)

        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)

            corrupt_img =y

            mask = None

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train (self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        real_net = DDP(self.net_real, device_ids=[opt.device])
        fake_net = DDP(self.net_fake, device_ids=[opt.device])

        ema = self.ema
        ema_fake = self.ema_fake

        optimizer, sched = build_optimizer_sched(opt, net, log,extra_params=None)
        fake_optimizer, fsched = build_fake_optimizer_sched(opt, fake_net, log=self.log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader = util.setup_loader(val_dataset, opt.microbatch)

        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        for it in range(opt.num_itr):
            net.train()
            fake_net.train()

            total_loss = 0.0
            for _ in range(n_inner_loop):
                # ======= prepare batch =======
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                T = torch.full((x0.shape[0],), opt.interval - 1, device=opt.device, dtype=torch.long)

                # ======= optimize net =======
                optimizer.zero_grad()
                OneStep_score= net(x1, T, cond=cond)
                OneStep_patch = self.compute_pred_x0(T, x1, OneStep_score, clip_denoise=None)

                step = torch.randint(0, int(0.98 * opt.interval), (x0.shape[0],), device=opt.device)

                xt,_,_= self.diffusion.q_sample(step, OneStep_patch, x1, ot_ode=opt.ot_ode)
                xt_label, alpha_t, beta_t=self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                v_t_label=self.compute_label(step,x0,xt_label)

                with torch.no_grad():

                    score_real= real_net(xt, step, cond=cond)
                    score_fake= fake_net(xt, step, cond=cond)

                time_weights= (1.0-alpha_t)/beta_t

                score_gradient = time_weights*(alpha_t*score_real+(1-alpha_t)*v_t_label - score_fake)

                score_gradient = torch.nan_to_num(score_gradient, nan=0.0)

                target = (OneStep_patch - score_gradient).detach()
                
                VSD_loss = 0.5*F.mse_loss(OneStep_patch, target)

                loss=VSD_loss
                loss.backward()
                optimizer.step()
                ema.update()

                # ======= optimize fake_net =======
                fake_optimizer.zero_grad()

                with torch.no_grad():
                    OneStep_score= net(x1, T, cond=cond)
                    OneStep_patch = self.compute_pred_x0(T, x1, OneStep_score, clip_denoise=None)

                step_fake = torch.randint(0, opt.interval, (OneStep_patch.shape[0],), device=opt.device)
                xt_fake,_,_= self.diffusion.q_sample(step_fake, OneStep_patch, x1, ot_ode=opt.ot_ode)
                v_t_fake=self.compute_label(step_fake,OneStep_patch,xt_fake)
                pred= fake_net(xt_fake, step_fake, cond=None)
                loss_fake = F.mse_loss(pred, v_t_fake)
                loss_fake.backward()
                fake_optimizer.step()
                ema_fake.update()

                total_loss += loss.item()

            # ============ scheduler step ============
            if sched is not None:
                sched.step()
            if fsched is not None:
                fsched.step()

            # ============ logging ============
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                it + 1,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(total_loss / n_inner_loop),
            ))

            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', total_loss / n_inner_loop)

            # ============ save checkpoint ============
            if it % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            # ============ evaluation ============
            if it == 200 or it % 200 == 0:
                net.eval()
                fake_net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
                fake_net.train()

        self.writer.close()


    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0
        
    @torch.no_grad()
    def Onestep_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        step_tensor=torch.full((x1.shape[0],), opt.interval-1, device=opt.device, dtype=torch.long)

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        def input_constructor(input_res):
            return dict(x=x1, steps=step_tensor, cond=cond)
        
        with self.ema.average_parameters():
            self.net.eval()
            self.net_real.eval()
            out= self.net(x1, step_tensor, cond=cond)
            val_onestep=self.compute_pred_x0(step_tensor, x1, out, clip_denoise=clip_denoise)
            conference=self.net_real(x1, step_tensor, cond=cond)
            conference_onestep=self.compute_pred_x0(step_tensor, x1, conference, clip_denoise=clip_denoise)
        return val_onestep.to(opt.device),conference_onestep.to(opt.device)


    def ddpm_dps_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_dps_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0


    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)

        xs,x_org= self.Onestep_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )#xs is the real perdiction, pred_x0s is the re-parametered result

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        x_org          = all_cat_cpu(opt, log, x_org)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid(img, nrow=nrow)) # [1,1] -> [0,1]

        log.info("Logging images ...")
        img_recon = xs
        log_image("image/clean",   img_clean[:,:,:,16,:])
        log_image("image/corrupt", img_corrupt[:,:,:,16,:])
        log_image("image/recon",   img_recon[:,:,:,16,:])
        log_image("image/org",   x_org[:,:,:,16,:])
        psnr_batch=[]
        ssim_batch=[]
        psnr_org_batch=[]
        ssim_org_batch=[]
        for i in range(img_recon.shape[0]):
            volume=img_recon.squeeze(dim=1).numpy()[i, :, :, :]
            org_img=x_org.squeeze(dim=1).numpy()[i, :, :, :]
            HR_image=img_clean.squeeze(dim=1).numpy()[i, :, :, :]
            psnr = peak_signal_noise_ratio(HR_image, volume, data_range=HR_image.max())
            ssim=structural_similarity(HR_image, volume, data_range=HR_image.max())
            psnr_org = peak_signal_noise_ratio(HR_image, org_img, data_range=HR_image.max())
            ssim_org=structural_similarity(HR_image, org_img, data_range=HR_image.max())
            psnr_batch.append(psnr)
            ssim_batch.append(ssim)
            psnr_org_batch.append(psnr_org)
            ssim_org_batch.append(ssim_org)
            average_psnr=np.mean(psnr_batch)
            average_ssim=np.mean(ssim_batch)
            average_org_psnr=np.mean(psnr_org_batch)
            average_org_ssim=np.mean(ssim_org_batch)
        print('test_psnr:{:.3f}\ttest_ssim:{:.4f}'.format(average_psnr, average_ssim))
        print('test_org_psnr:{:.3f}\ttest_org_ssim:{:.4f}'.format(average_org_psnr, average_org_ssim))

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
