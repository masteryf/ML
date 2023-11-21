import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler





class ECGDataset(Dataset):
    def __init__(self, dataframe, sequence_length=1000):
        # 移除列名中的额外引号
        dataframe.columns = dataframe.columns.str.strip("'")

        self.data = dataframe
        self.feature_columns = ['MLII', 'V5', 'V1', 'V4', 'V2']
        self.label_columns = ['S', 'x', 'A', 'R', 'F', 'j', 'e', 'N', '|', 'a', 'J', '+', '!', '~', ']', '/',
                              '[', 'Q', 'L', 'E', 'f', 'V']  # 移除 '""'
        self.sequence_length = sequence_length

        # 初始化标准化器
        self.feature_scaler = StandardScaler()

        # 用于计算均值和标准差，并对特征进行标准化
        self.feature_scaler.fit(self.data[self.feature_columns].values)

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = (idx + 1) * self.sequence_length

        # 获取并标准化特征和标签
        features = torch.tensor(self.feature_scaler.transform(self.data.iloc[start_idx:end_idx][self.feature_columns].values), dtype=torch.float32).t()
        labels = torch.tensor(self.data.iloc[start_idx:end_idx][self.label_columns].values, dtype=torch.float32).t()

        return features, labels







class fakemodel(nn.Module):


    def __init__(self):
        super(fakemodel, self).__init__()
        self.fc1 = nn.Linear(5000, 5120)
        self.bn1 = nn.BatchNorm1d(5120)
        self.fc2 = nn.Linear(5120, 2560)
        self.bn2 = nn.BatchNorm1d(2560)
        self.fc3 = nn.Linear(2560, 23000)
    def forward(self, x):
        # 假设df是通过Pandas读取的DataFrame
        df = pd.read_csv('fake.csv')

        # 创建数据集实例
        dataset = ECGDataset(df, sequence_length=1000)

        # 获取第一个样本
        sample_features, sample_labels = dataset[0]
        return sample_labels