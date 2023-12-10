import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class PupilsDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = "trainval",
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        self._split = verify_str_arg(split, "split", ("train", "test"))

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.base_folder = pathlib.Path(self.root)
        self.images_folder = self.base_folder / "images"
        self.masks_folder = self.base_folder / "masks"
        self.images = [i for i in os.listdir(self.images_folder) if os.path.splitext(i)[1] in ['.png']]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path = os.path.join(self.images_folder, self.images[idx])
        mask_path = os.path.join(self.masks_folder, self.images[idx])
        image = Image.open(img_path).convert("I")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


class PupilDatasetAugmented(PupilsDataset):
    def __init__(
            self,
            root: str,
            split: str,
            pre_transform=None,
            post_transform=None,
            pre_target_transform=None,
            post_target_transform=None,
            common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)

        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        # end if

        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return input, target