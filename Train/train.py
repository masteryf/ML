import torch.nn.functional as F


import time

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time() # 记录开始时间
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output, target = output.view(output.size(0), -1), target.view(target.size(0), -1)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    end_time = time.time() # 记录结束时间
    print("Training time for epoch {}: {:.2f} seconds".format(epoch, end_time - start_time)) # 计算并打印用时
