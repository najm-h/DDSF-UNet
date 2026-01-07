import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.cuda.amp import autocast
from core.losses import ssim_torch_safe, tv_l1, fft_mag_l1
import torchvision.utils as vutils
import torch.nn as nn

# =========================
# Train one epoch
# =========================

def train_one_epoch(model, loader, optimizer, scaler, epoch, device, vgg_perc,
                    use_amp=True, clip_grad=1.0):
    model.train()
    meter = 0.0
    count = 0
    amp_fw = autocast() if (use_amp and device == 'cuda') else nullcontext()

    for i, (inp, ref) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        ref = ref.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp_fw:
            out = model(inp)

        with autocast(enabled=False):
            out32 = out.clamp(0, 1).float()
            ref32 = ref.clamp(0, 1).float()

            # compute losses
            L1 = F.l1_loss(out32, ref32)
            Lssim = 1.0 - ssim_torch_safe(out32, ref32)
            Lperc = F.mse_loss(vgg_perc(out32), vgg_perc(ref32))
            Ltv = tv_l1(out32)
            # New frequency loss
            Lfreq = fft_mag_l1(out32, ref32)
            # Total loss
            loss = L1 + 0.1 * Lssim + 0.05 * Lperc + 0.01 * Ltv + 0.01 * Lfreq

        if not torch.isfinite(loss):
            print(f"[WARN e{epoch+1:03d}] Non-finite loss: {loss.item():.3f}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if clip_grad and clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        bs = inp.size(0)
        meter += loss.item() * bs
        count += bs

        if i == 0:
            os.makedirs('train_vis_path', exist_ok=True)
            os.makedirs('train_vis_ddsf_freq', exist_ok=True)
            K = min(bs, 2)
            for j in range(K):
                row = torch.stack(
                    [inp[j].cpu(), out[j].detach().cpu(), ref[j].cpu()],
                    0
                )
                vutils.save_image(
                    row,
                    f'train_vis_ddsf_freq/triplet_e{epoch+1}_idx{j}.png',
                    nrow=3,
                    normalize=False
                )

    return meter / max(1, count)
