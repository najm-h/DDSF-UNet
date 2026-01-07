import torch
import torch.nn.functional as F

def tv_l1(img):
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()

def ssim_torch_safe(x, y, C1=0.01**2, C2=0.03**2, ws=11, eps=1e-6):
    x = x.float()
    y = y.float()
    pad = ws // 2
    mx = F.avg_pool2d(x, ws, 1, pad)
    my = F.avg_pool2d(y, ws, 1, pad)
    x2 = F.avg_pool2d(x * x, ws, 1, pad)
    y2 = F.avg_pool2d(y * y, ws, 1, pad)
    xy = F.avg_pool2d(x * y, ws, 1, pad)
    sx = torch.clamp(x2 - mx * mx, min=0.0)
    sy = torch.clamp(y2 - my * my, min=0.0)
    sxy = xy - mx * my
    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx * mx + my * my + C1) * (sx + sy + C2)
    den = torch.clamp(den, min=eps)
    ssim_map = torch.clamp(num / den, -1.0, 1.0)
    return ssim_map.mean()


def fft_mag_l1(x, y):
    X = torch.fft.fft2(x, dim=(-2, -1))
    Y = torch.fft.fft2(y, dim=(-2, -1))
    magX = torch.abs(X)
    magY = torch.abs(Y)
    return F.l1_loss(magX, magY)