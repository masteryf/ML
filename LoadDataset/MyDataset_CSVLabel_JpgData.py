import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Tuple
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, file, start, end) -> None:
        self.data = file
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self._samples = [
            (
                torch.tensor(cv2.imread('LoadDataset/'+str(i)+'.jpg')),  # .reshape(48, 48),
                int(file["emotion"][i]),
            )
            for i in range(start, end)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())
        image = self.transform(image)
        return image, target