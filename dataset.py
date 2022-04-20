import glob
from collections.abc import Iterable
import os
import random
from typing import Optional, Callable

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms_: Optional[transforms.Compose] = None,
        unaligned: bool = True,
        mode: str = 'train',
        grayscale: bool = False
    ) -> None:
        self.transform: Callable[Image, torch.Tensor] = transforms.Compose(transforms_)
        self.unaligned: bool = unaligned
        self.grayscale: bool = grayscale

        self.files_A: Iterable[str] = sorted(glob.glob(os.path.join(root, f'{mode}/A') + '/*.*'))
        self.files_B: Iterable[str] = sorted(glob.glob(os.path.join(root, f'{mode}/B') + '/*.*'))

    def __getitem__(self, index: int):
        idx_A: int = index % len(self.files_A)
        item_A: Image = Image.open(self.files_A[idx_A])

        idx_B: int = random.randint(0, len(self.files_B) - 1) \
            if self.unaligned else (index % len(self.files_B))
        item_B: Image = Image.open(self.files_B[idx_B])
        
        if self.grayscale:
            item_A = item_A.convert('L')
            item_B = item_B.convert('L')

        return dict(A=self.transform(item_A), B=self.transform(item_B))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
