import os
import pandas as pd

# 指定CSV文件所在的文件夹路径
folder_path = 'D:\\Projects\\Datasets\\file'

# 获取文件夹中所有CSV文件的文件名
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 初始化一个空的DataFrame，用于存储合并后的数据
merged_data = pd.DataFrame()

# 逐个读取并合并CSV文件
for csv_file in csv_files:
    # 构建每个CSV文件的完整路径
    file_path = os.path.join(folder_path, csv_file)

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 将数据追加到合并后的DataFrame中
    merged_data = pd.concat([merged_data, data], ignore_index=True)

# 将合并后的数据写入新的CSV文件
merged_data.to_csv('1.csv', index=False)

print("合并完成！")