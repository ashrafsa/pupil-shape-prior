import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence

from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM

from SegNet.dataset import OxfordIIITPetsAugmented
from SegNet.model import ImageSegmentationDSC
from SegNet.pupils_dataset import PupilsDataset, PupilDatasetAugmented
from SegNet.utils import working_dir, get_device, to_device, print_test_dataset_masks, \
    t_custom_iou_loss, IoULoss, save_model_checkpoint, print_model_parameters, t2img, trimap2f, args_to_dict, ToDevice, \
    tensor_trimap


# Train the model for a single epoch
def train_model(model, loader, optimizer):
    to_device(model.train())
    cel = True
    if cel:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = IoULoss(softmax=True)
    # end if

    running_loss = 0.0
    running_samples = 0

    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)

        # The ground truth labels have a channel dimension (NCHW).
        # We need to remove it before passing it into
        # CrossEntropyLoss so that it has shape (NHW) and each element
        # is a value representing the class of the pixel.
        if cel:
            targets = targets.squeeze(dim=1)
        # end if
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_samples += targets.size(0)
        running_loss += loss.item()
    # end for

    print("Trained {} samples, Loss: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx + 1),
    ))


# Define training loop. This will train the model for multiple epochs.
#
# epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
#         The model is trained for [epoch[0] .. epoch[1]) epochs.
#
def train_loop(model, loader, test_data, epochs, optimizer, scheduler, save_path):
    test_inputs, test_targets = test_data
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)
        with torch.inference_mode():
            # Display the plt in the final training epoch.
            print_test_dataset_masks(model, test_inputs, test_targets, epoch=epoch, save_path=save_path,
                                     show_plot=(epoch == epoch_j - 1))
        # end with

        if scheduler is not None:
            scheduler.step()
        # end if
        print("")
    # end for


# ------ Validation: Check if CUDA is available
print(f"CUDA: {torch.cuda.is_available()}")
pupils_path_train = os.path.join(working_dir, 'source', 'train')
pupils_path_test = os.path.join(working_dir, 'source', 'test')
pupils_train_orig = PupilsDataset(root=pupils_path_train, split="trainval")
pupils_test_orig = PupilsDataset(root=pupils_path_test, split="test")

(train_input, train_target) = pupils_train_orig[0]

# train_input.show()


# Spot check a segmentation mask image after post-processing it
# via trimap2f().
t2img(trimap2f(train_target)).show()

# Create the train and test instances of the data loader for the
# Oxford IIIT Pets dataset with random augmentations applied.
# The images are resized to 128x128 squares, so the aspect ratio
# will be chaged. We use the nearest neighbour resizing algorithm
# to avoid disturbing the pixel values in the provided segmentation
# mask.
transform_dict = args_to_dict(
    pre_transform=T.ToTensor(),
    pre_target_transform=T.ToTensor(),
    common_transform=T.Compose([
        ToDevice(get_device()),
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
        # Random Horizontal Flip as data augmentation.
        T.RandomHorizontalFlip(p=0.5),
    ]),
    post_transform=T.Compose([
        # Color Jitter as data augmentation.
        T.ColorJitter(contrast=0.3),
    ]),
    post_target_transform=T.Compose([
        T.Lambda(tensor_trimap),
    ]),
)
train = PupilDatasetAugmented(root=pupils_path_train, split="trainval", **transform_dict)
test = PupilDatasetAugmented(root=pupils_path_test, split="test", **transform_dict)
train_loader = DataLoader(train, batch_size=64, shuffle=True, )
test_loader = DataLoader(test, batch_size=21, shuffle=True, )

(train_inputs, train_targets) = next(iter(train_loader))
(test_inputs, test_targets) = next(iter(test_loader))
print(f'{train_inputs.shape}, {train_targets.shape}')
print(t_custom_iou_loss())

# Run the model once on a single input batch to make sure that the model
# runs as expected and returns a tensor with the expected shape.
model_dsc = ImageSegmentationDSC(kernel_size=3)
model_dsc.eval()
to_device(model_dsc)
print(model_dsc(to_device(train_inputs)).shape)

# Check if our helper functions work as expected and if the image
# is generated as expected.
save_path = os.path.join(working_dir, "segnet_dsc_images")
os.makedirs(save_path, exist_ok=True)
print_test_dataset_masks(model_dsc, test_inputs, test_targets, epoch=0, save_path=None, show_plot=True)

to_device(model_dsc)
optimizer2 = torch.optim.Adam(model_dsc.parameters(), lr=0.001)
scheduler2 = None

# Train the model that uses depthwise separable convolutions.
train_loop(model_dsc, train_loader, (test_inputs, test_targets), (1, 21), optimizer2, scheduler2, save_path)