import torch
from torch import nn
from torch.nn import functional as F


class FCN1D(nn.Module):
    def __init__(self):
        in_channel = 3
        num_class = 20
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, 10, 101, 1, 50)
        self.conv2 = nn.Conv1d(10, 10, 21, 1, 10)
        self.final_conv = nn.Conv1d(10, num_class, 1)
        self.transpose_conv = nn.ConvTranspose1d(num_class, num_class, 11, 1, 5)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.final_conv(out)
        out = F.relu(out)
        out = self.transpose_conv(out)
        return out
