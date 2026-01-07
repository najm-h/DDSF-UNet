import random, math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from core.model import *
from core.losses import *
from core.dataset import *
from core.train_one_epoch import train_one_epoch
from utils.metrics import quick_eval
from utils.checkpoints import save_checkpoint
from utils.helpers import *


# =========================
# Reproducibility + Speed
# =========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------------------
# CLI arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train DDSF_UNet for underwater image enhancement")
    # Paths to datasets
    parser.add_argument("--raw_dir", type=str,
                        default="./dataset/raw",
                        help="Path to degraded images")
    parser.add_argument("--ref_dir", type=str,
                        default="./dataset/ref",
                        help="Path to reference images")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--crop", type=int, default=256, help="Random crop size")
    parser.add_argument("--base", type=int, default=32, help="Base number of channels")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)


    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = (device == 'cuda')
    print("device:", device)

    raw_list, ref_list = build_paired_lists(args.raw_dir, args.ref_dir)
    dataset = PairedUWDataset(raw_list, ref_list, crop=args.crop)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )

    eval_loader = loader  # replace with a true val loader if you have a split

    model = DDSF_UNet(base=args.base).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = GradScaler(enabled=use_amp)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    vgg_perc = PerceptualVGG19().to(device).eval()

    print("total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    best = float('inf')
    history = []

    for e in range(args.epochs):
        loss = train_one_epoch(
            model, loader, optimizer, scaler, e, device, vgg_perc,
            use_amp=use_amp, clip_grad=args.clip_grad
        )
        scheduler.step(loss if math.isfinite(loss) else best)
        history.append(float(loss))
        print(f"Epoch {e+1}/{args.epochs} - Loss: {loss:.6f}")

        is_best = math.isfinite(loss) and (loss < best)
        if is_best:
            best = loss

        save_checkpoint(
            {
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if use_amp else {},
                'best_loss': best
            },
            is_best,
            ckpt_dir=args.save_dir,
            freq=25
        )

        if (e + 1) % 10 == 0:
            P, S = quick_eval(model, eval_loader, device)
            print(f"[Eval @e{e+1}] PSNR={P:.2f}dB  SSIM={S:.4f}")

    print("best loss:", best)


if __name__ == "__main__":
    main()