import torch

def fftshift(x, dims=None):
    if dims is None:
        dims = tuple(range(x.ndim))
    shift = [d // 2 for d in x.shape[-len(dims):]]
    return torch.roll(x, shifts=shift, dims=dims)

def ifftshift(x, dims=None):
    if dims is None:
        dims = tuple(range(x.ndim))
    shift = [-(d // 2) for d in x.shape[-len(dims):]]
    return torch.roll(x, shifts=shift, dims=dims)

def real_fft3d_with_shift(x):
    x_freq = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')  # shape: (b, c, h, w, d//2+1)

    x_freq = fftshift(x_freq, dims=(-3, -2))

    return x_freq

def real_ifft3d_with_shift(x_freq,original_shape):
    x_freq = ifftshift(x_freq, dims=(-3, -2))

    x = torch.fft.irfftn(x_freq, s=original_shape[-3:], dim=(-3, -2, -1), norm='ortho')
    return x

def generate_center_sphere_mask(shape, radius):
    b, c, h, w, d = shape
    mask = torch.zeros((b, c, h, w, d), dtype=torch.float32)

    zz, yy, xx = torch.meshgrid(
        torch.arange(d),
        torch.arange(w),
        torch.arange(h),
        indexing='ij'  
    )

    center = torch.tensor([h // 2, w // 2, d // 2])
    dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2

    sphere = (dist_sq <= radius ** 2).float() 

    sphere = sphere.permute(2, 1, 0)  

    mask[:] = sphere.unsqueeze(0).unsqueeze(0)

    return mask
