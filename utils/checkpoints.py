import os
import shutil
import torch

# =========================
# Checkpoint saving
# =========================
def save_checkpoint(state, is_best, ckpt_dir='./checkpoints_Path', freq=25):
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = os.path.join(ckpt_dir, 'model_tmp.pth.tar')
    torch.save(state, tmp)
    epoch = state['epoch']
    if epoch % freq == 0:
        shutil.copyfile(tmp, os.path.join(ckpt_dir, f"model_{epoch}.pth.tar"))
    if is_best:
        shutil.copyfile(tmp, os.path.join(ckpt_dir, 'model_best.pth.tar'))
