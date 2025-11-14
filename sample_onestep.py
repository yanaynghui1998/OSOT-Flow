# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb_onestep import Runner, download_ckpt
from i2sb_onestep import ckpt_util
from dataset import HCP_loader2_volume
# import colored_traceback.always
from ipdb import set_trace as debug
import pathlib
import nibabel as nib
import time
from dataset.IXI_loader2_volume import TorchLowResolution as LR_opt
from monai.metrics import SSIMMetric, PSNRMetric
RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        corrupt_method=LR_opt(factor=7)
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.permute(0,1,3,4,2))
        x1 = y.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, clean_img

def sliding_window_reconstruct(patch_list, volume_shape, patch_size, stride, use_gaussian=False, sigma=0.5):
    """
    patch_list: List[Tensor], each patch shape: (B, C, p_d, p_h, p_w)
    volume_shape: tuple (B, C, D, H, W)
    patch_size: tuple (p_d, p_h, p_w)
    stride: tuple (s_d, s_h, s_w)
    
    return: Tensor of shape (B, C, D, H, W)
    """
    B, C, D, H, W = volume_shape
    p_d, p_h, p_w = patch_size
    s_d, s_h, s_w = stride

    output = torch.zeros(volume_shape, dtype=torch.float32, device=patch_list[0].device)
    weight_map = torch.zeros_like(output)

    if use_gaussian:
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, p_d),
            torch.linspace(-1, 1, p_h),
            torch.linspace(-1, 1, p_w),
            indexing='ij'
        )
        dist = xx**2 + yy**2 + zz**2
        window = torch.exp(-dist / (2 * sigma ** 2))
        window = (window / window.max()).to(patch_list[0].device)[None, None, :, :, :]  # (1,1,p_d,p_h,p_w)
    else:
        window = torch.ones((1, 1, p_d, p_h, p_w), device=patch_list[0].device)

    ct = 0
    for z in range(0, D - p_d + 1, s_d):
        for y in range(0, H - p_h + 1, s_h):
            for x in range(0, W - p_w + 1, s_w):
                patch = patch_list[ct] * window  # shape: (B, C, p_d, p_h, p_w)
                output[:, :, z:z+p_d, y:y+p_h, x:x+p_w] += patch
                weight_map[:, :, z:z+p_d, y:y+p_h, x:x+p_w] += window
                ct += 1

    weight_map[weight_map == 0] = 1.0
    return output / weight_map

def split_volume_to_patches_3d(volume, patch_size, stride):
    """
    volume: Tensor of shape (B, C, D, H, W)
    patch_size: tuple (p_d, p_h, p_w)
    stride: tuple (s_d, s_h, s_w)
    
    return: List[Tensor], 每个 patch shape: (B, C, p_d, p_h, p_w)
    """
    B, C, D, H, W = volume.shape
    p_d, p_h, p_w = patch_size
    s_d, s_h, s_w = stride
    patch_list = []

    for z in range(0, D - p_d + 1, s_d):
        for y in range(0, H - p_h + 1, s_h):
            for x in range(0, W - p_w + 1, s_w):
                patch = volume[:, :, z:z+p_d, y:y+p_h, x:x+p_w]
                patch_list.append(patch)  # shape: (B, C, p_d, p_h, p_w)
    
    return patch_list

def compute_nmse(gt, pred):
    """
    gt: ground truth volume, shape (D, H, W)
    pred: predicted volume, shape (D, H, W)
    """
    return np.sum((pred - gt)**2) / np.sum(gt**2)

# @torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")
    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    corrupt_method =None
    LR=LR_opt(factor=7)
    val_dataset=HCP_loader2_volume.HCPVolumes(opt.dataset_dir, opt.dataset_dir,mode='val')
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    ys = []
    num = 0

    average_psnr, average_ssim, average_time, average_nmse = 0.0, 0.0, 0.0, 0.0
    batch_psnr, batch_ssim,batch_time, batch_nmse = [], [], [], []
    count=0
    for loader_itr, out in enumerate(val_loader):

        corrupt_img, x1, mask, cond, clean_img = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)
        xs_list=[]
        x1_patches=split_volume_to_patches_3d(x1,patch_size=(64,64,64),stride=(32,32,32))
        start_time = time.time()
        for _,x1_patch in enumerate(x1_patches):
            xs, _ = runner.Onestep_sampling(
            ckpt_opt, x1_patch, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1)
            xs_list.append(xs[:,:,:,:,:])
        infer_time = time.time() - start_time
        xs=sliding_window_reconstruct(xs_list, x1.shape, patch_size=(64,64,64), stride=(32,32,32), use_gaussian=True)
        xs=torch.clamp(xs,0,1).detach().cpu()
        xs.requires_grad_(True)
        '''
        data consistency
        '''
        xs=xs.permute(0,1,3,4,2)
        clean_img=clean_img.permute(0,1,3,4,2)
        xs_corrputed=LR(xs)
        residual=xs_corrputed-x1
        residual_norm = (torch.linalg.norm(residual) ** 2)
        norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xs)[0]
        xs_dc=xs-0.5*norm_grad
        xs_dc=torch.clamp(xs_dc,0,1)

        recon_img_dc=xs_dc.to(opt.device)

        psnr = psnr_metric(xs_dc, clean_img)
        ssim = ssim_metric(xs_dc, clean_img)

        assert recon_img_dc.shape == x1.shape==clean_img.shape

        batch_psnr.append(psnr)
        batch_ssim.append(ssim)
        batch_time.append(infer_time)

        if len(recon_img_dc.shape) == 5:
            recon_img_dc=recon_img_dc.squeeze(dim=1)
            LR_volume=x1.squeeze(dim=1)
            clean_img=clean_img.squeeze(dim=1)

        pathlib.Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

        for i in range(recon_img_dc.shape[0]): 
            recon_img_dc=recon_img_dc.detach().cpu().numpy()[i, :, :, :]
            LR_volume=LR_volume.detach().cpu().numpy()[i, :, :, :]
            clean_img=clean_img.detach().cpu().numpy()[i, :, :, :]

            output_name = os.path.join(opt.output_dir, f'sample_{count}.nii.gz')

            output_name = os.path.join(opt.output_dir, f'rec_sample_dc_{count}.nii.gz')
            img_dc = nib.Nifti1Image(recon_img_dc, np.eye(4))
            nib.save(img=img_dc, filename=output_name)

            output_name_LR = os.path.join(opt.output_dir, f'LR_sample_{count}.nii.gz')
            img_LR = nib.Nifti1Image(corrupt_img, np.eye(4))
            nib.save(img=img_LR, filename=output_name_LR)

            output_name_HR = os.path.join(opt.output_dir, f'HR_sample_{count}.nii.gz')
            img_HR = nib.Nifti1Image(clean_img, np.eye(4))
            nib.save(img=img_HR, filename=output_name_HR)

            print(f'Saved to {output_name}')
        count+=1
    average_psnr=np.mean(batch_psnr)
    std_psnr=np.std(batch_psnr,ddof=1)
    average_ssim=np.mean(batch_ssim)
    std_ssim=np.std(batch_ssim,ddof=1)
    average_time=np.mean(batch_time)

    print('average_time:{:.3f}s\ttest_psnr:{:.3f}\ttest_ssim:{:.4f}'.format(
        average_time, average_psnr, average_ssim))
    
    print('std_psnr:{:.3f}\tstd_ssim:{:.3f}'.format(
            std_psnr, std_ssim))
    
    with open("test_results.txt", "w") as f:
        f.write('average_time:{:.3f}s\ttest_psnr:{:.3f}\ttest_ssim:{:.4f}\n'.format(
            average_time, average_psnr, average_ssim))
        
        f.write('std_psnr:{:.3f}\tstd_ssim:{:.3f}\n'.format(
            std_psnr, std_ssim))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=64)
    parser.add_argument("--dataset-dir",    type=Path, default="",  help="path to LMDB dataset")
    parser.add_argument("--output-dir",    type=str, default="")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=1)
    parser.add_argument("--ckpt",           type=str,  default="",        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=10,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    # download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
