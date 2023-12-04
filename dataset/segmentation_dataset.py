import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class PupilsDataSet(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # TD refactoring to [x for x in p.iterdir() if x.is_file() and x.suffix in ['.png']]
        self.images = [i for i in os.listdir(
            image_dir) if os.path.splitext(i)[1] in ['.png']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        # image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        image = np.array(Image.open(img_path).convert("I"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"].unsqueeze(0)
        return image, mask