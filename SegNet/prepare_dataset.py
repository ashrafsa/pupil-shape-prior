import json
import os
import shutil

import torch
from torch.utils.data.dataset import random_split, Subset

from SegNet.pupils_dataset import PupilsDataset
from unet.instance_dataset import PupilsInstanceDataset


def split(full_dataset, val_percent, test_percent, random_seed=None) -> tuple[Subset, Subset, Subset]:
    amount = len(full_dataset)

    test_amount = (int(amount * test_percent) if test_percent is not None else 0)
    val_amount = (int(amount * val_percent) if val_percent is not None else 0)
    train_amount = amount - test_amount - val_amount

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        (train_amount, val_amount, test_amount),
        generator=(torch.Generator().manual_seed(random_seed) if random_seed else None))

    return train_dataset, val_dataset, test_dataset


ROOT = "../dataset/source"
DST = "../dataset/PupilsData"

folders = dict({
    'train_images': os.path.join(DST, 'train/images'),
    'train_masks': os.path.join(DST, 'train/masks'),
    'test_images': os.path.join(DST, 'test/images'),
    'test_masks': os.path.join(DST, 'test/masks'),
})

for k in folders:
    folder = folders[k]
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    pass

images_path = os.path.join(ROOT, 'images')
masks_path = os.path.join(ROOT, 'masks')
images_names = os.listdir(images_path)
train_ds, val_ds, test_ds = split(images_names, 0, 0.1, 42)

for item in train_ds:
    src_img = os.path.join(images_path, item)
    src_mask = os.path.join(masks_path, item)
    train_img = os.path.join(folders['train_images'], item)
    shutil.copy(src_img, train_img)
    train_mask = os.path.join(folders['train_masks'], item)
    shutil.copy(src_mask, train_mask)
    print(item)

for item in test_ds:
    src_img = os.path.join(images_path, item)
    src_mask = os.path.join(masks_path, item)
    train_img = os.path.join(folders['test_images'], item)
    shutil.copy(src_img, train_img)
    train_mask = os.path.join(folders['test_masks'], item)
    shutil.copy(src_mask, train_mask)
    print(item)