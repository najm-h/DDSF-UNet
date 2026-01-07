import os
import shutil
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from core.model import *
from core.losses import *
from core.dataset import *
from core.train_one_epoch import train_one_epoch


# =========================
# Metrics (batch-safe)
# =========================

@torch.no_grad()
def compute_metrics_batch(output: torch.Tensor, target: torch.Tensor, return_lists: bool = False):
    output = output.clamp(0, 1).detach().cpu().float()
    target = target.clamp(0, 1).detach().cpu().float()
    B = output.size(0)
    psnrs, ssims = [], []
    for i in range(B):
        o = output[i].permute(1, 2, 0).numpy()
        t = target[i].permute(1, 2, 0).numpy()
        psnrs.append(psnr(t, o, data_range=1.0))
        ssims.append(ssim(t, o, data_range=1.0, channel_axis=2))
    if return_lists:
        return psnrs, ssims
    return float(np.mean(psnrs)), float(np.mean(ssims))



@torch.no_grad()
def quick_eval(model, loader, device):
    model.eval()
    P, S = [], []
    for inp, ref in loader:
        inp, ref = inp.to(device), ref.to(device)
        out = model(inp)
        p_list, s_list = compute_metrics_batch(out, ref, return_lists=True)
        P.extend(p_list)
        S.extend(s_list)
    return float(np.mean(P)), float(np.mean(S))
