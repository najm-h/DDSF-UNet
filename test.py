import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import torch
from torchvision import transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.helpers import *


# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Test Underwater Image Enhancement Model")
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/model_best.pth.tar")
parser.add_argument("--raw_test_dir", type=str, default="./dataset/test/raw")
parser.add_argument("--ref_test_dir", type=str, default="./dataset/test/ref")
parser.add_argument("--out_dir", type=str, default="./test_results")
parser.add_argument("--divisor", type=int, default=8)

args = parser.parse_args()

# ----------------------------
# PATHS
# ----------------------------
ckpt_path = args.ckpt_path
raw_test_dir = args.raw_test_dir
ref_test_dir = args.ref_test_dir
out_dir = args.out_dir
divisor = args.divisor

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# Utils
# ----------------------------
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# ----------------------------
# Utils
# ----------------------------
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# ----------------------------
# Build model + load ckpt
# ----------------------------

def main():

    model, base = build_and_load_model(ckpt_path, device)
    # ----------------------------
    # Pair files by stem
    # ----------------------------
    raw_map = {Path(f).stem: os.path.join(raw_test_dir, f) for f in list_imgs(raw_test_dir)}
    ref_map = {Path(f).stem: os.path.join(ref_test_dir, f) for f in list_imgs(ref_test_dir)}
    keys = sorted(set(raw_map) & set(ref_map))

    print("Raw files:", len(raw_map), "| Ref files:", len(ref_map), "| Pairs:", len(keys))
    if len(keys) == 0:
        raise RuntimeError("No paired filenames found between raw_test_dir and ref_test_dir. Check naming.")

    # ----------------------------
    # Infer + Save + Metrics
    # ----------------------------
    psnrs, ssims = [], []

    with torch.inference_mode():
        for k in tqdm(keys, desc="Infer+Eval"):
            with Image.open(raw_map[k]) as im_raw:
                raw = im_raw.convert("RGB")
            with Image.open(ref_map[k]) as im_ref:
                ref = im_ref.convert("RGB")

            x = to_tensor(raw).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
            x_pad, ph, pw = pad_to_divisible(x, 8)

            y = model(x_pad)

            # IMPORTANT: handle tuple/list outputs
            if isinstance(y, (tuple, list)):
                y = y[0]

            y = unpad(y, ph, pw).clamp(0, 1).squeeze(0).cpu()  # [3,H,W]

            # save output
            to_pil(y).save(os.path.join(out_dir, f"{k}_enhanced.png"))

            # compute metrics
            y_np = y.permute(1, 2, 0).numpy().astype(np.float32)      # HWC [0,1]
            ref_np = (np.asarray(ref, dtype=np.float32) / 255.0)      # HWC [0,1]

            # strict check (since you said test_input and test_gt are same size)
            if y_np.shape != ref_np.shape:
                raise RuntimeError(f"Shape mismatch: pred {y_np.shape} vs gt {ref_np.shape} for {k}")

            psnrs.append(psnr(ref_np, y_np, data_range=1.0))

            try:
                ssims.append(ssim(ref_np, y_np, data_range=1.0, channel_axis=2))
            except TypeError:
                ssims.append(ssim(ref_np, y_np, data_range=1.0, multichannel=True))

    print(f"TEST Avg PSNR: {float(np.mean(psnrs)):.2f} dB")
    print(f"TEST Avg SSIM: {float(np.mean(ssims)):.4f}")
    print("Saved outputs to:", out_dir)
    print("Device:", device, "| Loaded base:", base)


if __name__ == "__main__":
    main()