import torch
import torchvision
from torch.utils.data import DataLoader

from dataset.segmentation_dataset import PupilsDataSet


def get_loaders(train_dir,
                train_mask_dir,
                val_img_dir,
                val_mask_dir,
                batch_size,
                train_transform,
                val_transform,
                num_workers=4,
                pin_memory=True):
    train_ds = PupilsDataSet(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_ds = PupilsDataSet(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader


def save_predictions_as_imgs(loader, model, folder='saved_images/', device='cuda'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > .5).float()
            preds[preds == 1.0] = 255.0
        torchvision.utils.save_image(preds, f'{folder}/{idx}_pred.png')
        torchvision.utils.save_image(y, f'{folder}/{idx}_target.png')
    model.train()


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > .5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    acc = num_correct / num_pixels * 100
    dice_score = dice_score / len(loader) * 100
    print(f'Got {num_correct}/{num_pixels} with acc = {acc:.3f}')
    print(f'Dice score: {dice_score:.3f}')
    model.train()
    return acc.item(), dice_score.item()