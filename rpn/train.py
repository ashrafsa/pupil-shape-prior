import torch

from rpn.pupils_dataset import PupilsDataset
from rpn.rpn_utils import train_val_dataset

dataset = PupilsDataset('../../Data/PennFudanPed', None)

datasets = train_val_dataset(dataset)
print(len(datasets['train']))
print(len(datasets['val']))

data_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=15, shuffle=True, num_workers=2,
                                          collate_fn=collate_fn)
data_loader_val = torch.utils.data.DataLoader(datasets['val'], batch_size=15, shuffle=True, num_workers=2,
                                              collate_fn=collate_fn)