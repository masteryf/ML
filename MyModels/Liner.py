import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(4, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc3(x))
        return x

class Gnet(nn.Module):
    def __init__(self, opt):
        super(Gnet, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1024)
        #input = (n, c, h, w = 256, 100, 1, 1)
        #output = (n, c, h, w = 256, 1, 32, 32)


    def forward(self, x):
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(32, 32)
        return x

class Dnet(nn.Module):
    def __init__(self, opt):
        super(Dnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)

    def forward(self, x):
        in_size = x.size(0)
        print(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2), 2)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
        return x