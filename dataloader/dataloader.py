import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, DatasetFolder


# class ImageLoader(DataLoader):
#     def __init__(self):
#         pass

class ImageLoader(DatasetFolder):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        # dataset=None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        # self.dataset = dataset

    def take_item(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img1, target1 = self.take_item(index)
        # take different cls
        second_idx = round(np.random.uniform(0, len(self.imgs)))

        if second_idx == index:
            return img1, target1

        lmd = round(np.random.beta(1, 1), 4)

        img2, target2 = self.take_item(second_idx)
        # return target1, target2
        mix_img = lmd * img1 + (1 - lmd) * img2
        mix_target = lmd * target1 + (1 - lmd) * target2 if torch.any(target1 != target2) else target1

        # print('ya tut')

        return mix_img, mix_target
