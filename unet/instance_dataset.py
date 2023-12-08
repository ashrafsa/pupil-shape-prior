import json
import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split


class PupilsInstanceDataset(Dataset):
    np.random.seed(0)
    torch.manual_seed(0)

    def __init__(self, img_dir, ann_dir=None, ext='.png'):
        if ann_dir is None:
            ann_dir = img_dir

        img_path = pathlib.Path(img_dir)
        ann_path = pathlib.Path(ann_dir)

        if (not img_path.exists() and not img_path.is_dir()) or \
                (not ann_path.exists() and not ann_path.is_dir()):
            raise NotADirectoryError

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_names = [i for i in os.listdir(img_dir) if
                          os.path.isfile(pathlib.Path(img_dir).joinpath(i)) and os.path.splitext(i)[1] == ext]
        self.ext = ext

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace(self.ext, '.json'))
        img = Image.open(img_path)
        ann = json.load(open(ann_path))
        mask = np.zeros(shape=(img.size[1], img.size[0]), dtype=np.uint8)
        for obj in ann['shapes']:
            if obj['label'] == 'pupil':
                poly = [(x, y) for x, y in obj['points']]
                poly = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [poly], 1)
        img = np.array(img)
        mask = np.expand_dims(mask, axis=0)
        sample = {'image': img, 'mask': mask, 'name': img_name}
        return sample

    def save_masks_of_all_images(self, target_path=r'mask'):
        full_path = pathlib.Path(self.img_dir + '_' + target_path)

        if not full_path.exists():
            full_path.mkdir()

        for item in self:
            # Mask saving
            maskc = np.squeeze(item['mask'], axis=0)
            maskc[maskc == 1] = 255

            mask_img = Image.fromarray(maskc)
            mask_file_name = full_path.joinpath(item['name'])
            mask_img.save(mask_file_name)