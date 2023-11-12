import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class GoogleLeNet(nn.Module):

    def __init__(self):
        super(GoogleLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.gle = torchvision.models.GoogLeNet()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.gle(x)
        return x