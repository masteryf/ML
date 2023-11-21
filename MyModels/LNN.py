import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(5000, 5120)
        self.bn1 = nn.BatchNorm1d(5120)
        self.fc2 = nn.Linear(5120, 2560)
        self.bn2 = nn.BatchNorm1d(2560)
        self.fc3 = nn.Linear(2560, 23000)

    def forward(self, x):
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc3(x))
        x = x.view(in_size, 23, -1)
        return x
