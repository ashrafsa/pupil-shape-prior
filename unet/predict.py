import os.path
import time

import cv2
import numpy as np
import torch
import torchvision
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2

import UnetModel
from train import IMAGE_HEIGHT, IMAGE_WIDTH
from utils import load_model, scale_contour, visualize_predicted_contour

device = 'cpu'
model = load_model(fr'C:\dev\pupil-shape-prior\models\unet_ps_400x300_Epoch335of500_Acc99.993_Dice90.141.pth.tar')
model.eval()

#folder = fr'C:\ProgramData\Shamir\Spark4w\Images\9514df2f-a3dc-40bd-b960-6df81369d902'
folder = fr'C:\dev\pupil-shape-prior\dataset\source\test_image'
image_path = os.path.join(folder, '20231104_100931_a20fb883_IR_RIGHT.png')
#image_path = os.path.join(folder, 'SENSOR_IR_RIGHT.png')
pil = Image.open(image_path).convert("I")
image = np.array(pil, dtype=np.float32)
# image = image[:, :, 0]
transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_CUBIC),
    A.Normalize(mean=0.456, std=0.224),
    ToTensorV2()
])
augmentations = transform(image=image.copy())
image = augmentations["image"]
image = image.unsqueeze(0)
t0 = time.time()
with torch.no_grad():
    predicted_mask = torch.sigmoid(model(image))
t1 = time.time()
print(f'{round(t1-t0):.2f} s')
predicted_mask = (predicted_mask > .5).float()
predicted_mask[predicted_mask == 1.0] = 255.0
#torchvision.utils.save_image(predicted_mask, os.path.join(folder, '_prediction.png'))
mask = predicted_mask.squeeze(0, 1)
mask = mask.byte().cpu().numpy()

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for c in contours:
    for p in c:
        p[0][0] = int(p[0][0] * pil.width / IMAGE_WIDTH)
        p[0][1] = int(p[0][1] * pil.height / IMAGE_HEIGHT)

print(contours)
img = cv2.imread(image_path)
visualize_predicted_contour(img, contours)