from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


# =========================
# Dataset
# =========================
class PairedUWDataset(Dataset):
    """Paired dataset with identical random crop + safe paired augments."""
    def __init__(self, raw_list, ref_list, crop=256):
        assert len(raw_list) == len(ref_list)
        self.raw_list = raw_list
        self.ref_list = ref_list
        self.crop = crop
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, idx):
        raw = Image.open(self.raw_list[idx]).convert('RGB')
        ref = Image.open(self.ref_list[idx]).convert('RGB')

        if self.crop is not None:
            w, h = raw.size
            if min(w, h) < self.crop:
                s = self.crop / min(w, h)
                new_w, new_h = int(round(w * s)), int(round(h * s))
                raw = raw.resize((new_w, new_h), Image.BICUBIC)
                ref = ref.resize((new_w, new_h), Image.BICUBIC)
                w, h = new_w, new_h
            x = random.randint(0, w - self.crop)
            y = random.randint(0, h - self.crop)
            box = (x, y, x + self.crop, y + self.crop)
            raw, ref = raw.crop(box), ref.crop(box)

        if random.random() < 0.5:
            raw = raw.transpose(Image.FLIP_LEFT_RIGHT)
            ref = ref.transpose(Image.FLIP_LEFT_RIGHT)
        k = random.randint(0, 3)
        if k == 1:
            raw = raw.transpose(Image.ROTATE_90)
            ref = ref.transpose(Image.ROTATE_90)
        elif k == 2:
            raw = raw.transpose(Image.ROTATE_180)
            ref = ref.transpose(Image.ROTATE_180)
        elif k == 3:
            raw = raw.transpose(Image.ROTATE_270)
            ref = ref.transpose(Image.ROTATE_270)

        return self.to_tensor(raw), self.to_tensor(ref)
