import torch
from MyModels.FCN import FCN1D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FCN1D().to(DEVICE)

x = torch.rand(size=(10, 5, 1000))

print(net(x).shape)