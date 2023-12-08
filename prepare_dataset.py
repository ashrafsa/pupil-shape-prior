import json
import os
import shutil

import torch
from torch.utils.data.dataset import random_split, Subset

from dataset.instance_dataset import PupilsInstanceDataset

ROOT = "./dataset/source"


def combine_labels(raw_images: str, raw_labels: str):
    images_folder: str = os.path.join(ROOT, raw_images)
    labels_folder = os.path.join(ROOT, raw_labels)
    # images are folders under the root data set
    images_list = os.listdir(images_folder)
    for item in images_list:
        jsonFile = os.path.join(images_folder, item, f'{item}.json')
        job = json.load(open(jsonFile))
        print(f'start {job["SessionID"]}...')
        for img in job['Images']:
            imagePath = os.path.join(images_folder, item, img)
            imageDir = os.path.dirname(imagePath)
            print(f'copy {imagePath} to {labels_folder}')
            shutil.copy(imagePath, labels_folder)
            pass
        print(f'{job["SessionID"]} done.')
        pass
    print(images_list)


# combine_labels("pupils_dataset_1", "images")
def save_masks():
    ds = PupilsInstanceDataset(os.path.join(ROOT, 'images'))
    print(ds.img_names)
    ds.save_masks_of_all_images()
    pass


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


train_dir = os.path.join(ROOT, 'train_image')
train_mask_dir = os.path.join(ROOT, 'train_mask')
val_dir = os.path.join(ROOT, 'val_image')
val_mask_dir = os.path.join(ROOT, 'val_mask')
test_dir = os.path.join(ROOT, 'test_image')

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.isdir(train_mask_dir):
    shutil.rmtree(train_mask_dir)
os.makedirs(train_mask_dir)

if os.path.isdir(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

if os.path.isdir(val_mask_dir):
    shutil.rmtree(val_mask_dir)
os.makedirs(val_mask_dir)

if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)

images_path = os.path.join(ROOT, 'images')
masks_path = os.path.join(ROOT, 'masks')
dataset = PupilsInstanceDataset(images_path)
train_ds, val_ds, test_ds = split(dataset.img_names, 0.2, 0.1, 50)

for item in train_ds:
    src_img = os.path.join(images_path, item)
    src_mask = os.path.join(masks_path, item)
    train_img = os.path.join(train_dir, item)
    shutil.copy(src_img, train_img)
    train_mask = os.path.join(train_mask_dir, item)
    shutil.copy(src_mask, train_mask)
    print(item)

for item in val_ds:
    src_img = os.path.join(images_path, item)
    src_mask = os.path.join(masks_path, item)
    val_img = os.path.join(val_dir, item)
    shutil.copy(src_img, val_img)
    val_mask = os.path.join(val_mask_dir, item)
    shutil.copy(src_mask, val_mask)
    print(item)

for item in test_ds:
    src_img = os.path.join(images_path, item)
    src_mask = os.path.join(masks_path, item)
    test_img = os.path.join(test_dir, item)
    shutil.copy(src_img, test_img)
    print(item)