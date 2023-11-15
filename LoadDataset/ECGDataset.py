import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, dataframe, sequence_length=1000):
        # 移除列名中的额外引号
        dataframe.columns = dataframe.columns.str.strip("'")

        self.data = dataframe
        self.feature_columns = ['MLII', 'V5', 'V1', 'V4', 'V2']
        self.label_columns = ['S', 'x', 'A', 'R', 'F', 'j', 'e', 'N', '|', 'a', 'J', '+', '!', '~', ']', '/',
                              '[', 'Q', 'L', 'E', 'f', 'V']  # 移除 '""'
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = (idx + 1) * self.sequence_length

        features = torch.tensor(self.data.iloc[start_idx:end_idx][self.feature_columns].values, dtype=torch.float32).t()
        labels = torch.tensor(self.data.iloc[start_idx:end_idx][self.label_columns].values, dtype=torch.float32).t()

        return features, labels

# # 示例用法
import pandas as pd

# 假设df是通过Pandas读取的DataFrame
df = pd.read_csv('D:\\Projects\\ML\\tools\\1.csv')

# 创建数据集实例
dataset = ECGDataset(df, sequence_length=1000)

# 获取第一个样本
sample_features, sample_labels = dataset[0]
print(sample_features.shape)  # 应该输出 torch.Size([5, 1000])
print(sample_labels.shape)    # 应该输出 torch.Size([23, 1000])
