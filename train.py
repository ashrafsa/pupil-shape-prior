import os
import shutil
import time

import cv2
import torch
from torch import nn, optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from UnetModel import UNet
from utils import get_loaders, save_predictions_as_imgs, save_checkpoint, check_accuracy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 10  # 7
NUM_EPOCHS = 500
NUM_WORKERS = 4
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 300
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_PATH = r'./dataset/source'
TRAIN_IMG_DIR = rf'{DATASET_PATH}/train_image'
TRAIN_MASK_DIR = rf'{DATASET_PATH}/train_mask'
VAL_IMG_DIR = rf'{DATASET_PATH}/val_image'
VAL_MASK_DIR = rf'{DATASET_PATH}/val_mask'
save_images = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        torch.cuda.empty_cache()
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        if DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            # Runs the forward pass with autocasting.
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

        # predictions = model(data)
        # loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # test
        del data
        del targets

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_CUBIC, ),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=0.456, std=0.224),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_CUBIC, ),
        A.Normalize(mean=0.456, std=0.224),
        ToTensorV2()
    ])

    # create instance of the model
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    #
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # data loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # scaler
    scaler = torch.cuda.amp.GradScaler()
    max_acc = max_dice = -999
    max_epoch_acc = 0
    max_epoch_dice = 0

    # loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check accuracy
        acc, dice = check_accuracy(val_loader, model, device=DEVICE)

        if max_dice < dice:
            max_epoch_dice = epoch
            max_dice = dice

        if max_acc < acc:
            max_epoch_acc = epoch
            max_acc = acc

        filename = fr'models/unet_ps_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_Epoch{epoch + 1}of{NUM_EPOCHS}_Acc{round(acc, 3)}_Dice{round(dice, 3)}.pth.tar'
        shutil.copy(r'checkpoint.pth.tar', filename)
        if save_images:
            folder = fr'saved_images/'
            folder = folder + fr'i_{IMAGE_HEIGHT}x{IMAGE_WIDTH}Epoch{epoch + 1}of{NUM_EPOCHS}/'
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            save_predictions_as_imgs(val_loader, model, device=DEVICE, folder=folder)

        end_time = time.time()
        print(
            f'\r\n Current maximal scores:\r\n\tMax Dice score is {max_dice:.4f} in epoch number: {max_epoch_dice + 1}')
        print(f'\tMax Accuracy score is {max_acc:.4f} in epoch number: {max_epoch_acc + 1}')
        print(f'\r\nEpoch #{epoch + 1} duration time is {round(end_time - start_time)} sec')
    print('|---------------------------------------------------------------------|')
    print(f'Max Dice score is {max_dice:.4f} in epoch number: {max_epoch_dice + 1}')
    print(f'Max Accuracy score is {max_acc:.4f} in epoch number: {max_epoch_acc + 1}')
    print('|---------------------------------------------------------------------|')


if __name__ == '__main__':
    main()