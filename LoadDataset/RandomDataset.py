import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Tuple


class MyDataset(Dataset):

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        data = torch.rand(size=(3, 1000))
        target = torch.rand(size=(20, 1000))
        return data, target
