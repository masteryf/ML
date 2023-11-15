import torch
import pandas as pd
from torch import optim
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from LoadDataset.ECGDataset import ECGDataset
from MyModels.FCN import FCN1D
from Test import test
from Train import train

print(torch.cuda.is_available())

BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNRATE = 0.00001

df: DataFrame = pd.read_csv('D:/Projects/Datasets/1.csv', encoding='utf-8')
#
# df = df.sample(frac=1)
cut_idx = int(round(0.2 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx].reset_index(drop=True), df.iloc[cut_idx:].reset_index(drop=True)
print(df)
Train = ECGDataset(df_train)
Test = ECGDataset(df_test)

train_loader = torch.utils.data.DataLoader(Train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(Test, shuffle=True)

model = FCN1D().to(DEVICE)
optimizer = optim.Adam(model.parameters())

for epoch in range(1, EPOCHS + 1):
    train.train(model, DEVICE, train_loader, optimizer, epoch)
    test.test(model, DEVICE, test_loader)
    # d'ftorch.save(MyModels.state_dict(), "MyModels/model_weight/Test.pth")
