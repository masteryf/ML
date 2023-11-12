import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Tuple
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, file, start, end) -> None:
        self.data = file
        self.transform = transforms.Compose([
            transforms.Resize([1, 5]),
            transforms.ToTensor(),
        ])
        self._samples = [
            (
                torch.tensor([int(idx) for idx in file["k"][i].split()], dtype=torch.uint8).reshape(1, 5),
                int(file["v"][i]),
            )
            for i in range(start, end)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())
        image = self.transform(image)
        print(image)
        return image, target