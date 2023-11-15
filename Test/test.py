import torch.nn.functional as F
import torch

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output, target = output.view(output.size(0), -1), target.view(target.size(0), -1)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 将一批的损失相加


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss, correct, len(test_loader.dataset)))