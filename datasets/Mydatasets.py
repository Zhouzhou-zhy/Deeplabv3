import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.multiprocessing
import numpy as np
torch.multiprocessing.set_sharing_strategy("file_system")


class MyDataset(Dataset):

    def dataset_cmap(N=256, normalized=False):
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap
    camp=dataset_cmap()
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(root, "image")))
        self.masks = sorted(os.listdir(os.path.join(root, "label")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "image", self.images[idx])
        mask_path = os.path.join(self.root, "label", self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image,mask = self.transform(image,mask)
           

        return image, mask

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        # 假设您有一个颜色映射表 cmap
        return cls.cmap[mask]
