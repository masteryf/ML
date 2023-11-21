import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # 导入PIL库
from ast import literal_eval
from utils.markovTrans import MarkovTrans
import torch.nn.functional as F


class ECGDataset(Dataset):
    label_map = {'e': 0, '|': 1, 'A': 2, 'Q': 3, 'f': 4, 'j': 5, '~': 6, 'E': 7, 'L': 8, '!': 9, 'F': 10, 'J': 11, 'R': 12, 'N': 13, 'a': 14, '/': 15, 'V': 16, 'S': 17, 'x': 18}

    def __init__(self, dataframe):
        self.data = dataframe
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data['Target'] = self.data['Target'].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tag = self.data.iloc[idx]['Tag']
        target = self.data.iloc[idx]['Target']
        tag = literal_eval(tag)
        image = MarkovTrans(tag)

        # 转换图像为PIL图像
        image = Image.fromarray(image)

        # 将输入图像调整为期望的输出维度
        image = self.transform(image)
        # target_one_hot = F.one_hot(torch.LongTensor([target]), num_classes=1000).squeeze(0)
        target_tensor = torch.LongTensor([target])  # 假设 'Target' 是一个标量，根据实际情况调整
        return image, target_tensor
