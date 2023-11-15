import torch
from MyModels.LNN import NN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = NN().to(DEVICE)

x = torch.rand(size=(10, 5, 1000))

print(net(x).shape)