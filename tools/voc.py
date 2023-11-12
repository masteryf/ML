import os

# 指定要扫描的目录
source_directory = "C:\\Users\\64783\\OneDrive\\文档\\Projects\\ML\\data\\dani\\val\\image"

# 指定保存文件名的文本文件路径
output_file_path = "C:\\Users\\64783\\OneDrive\\文档\\Projects\\ML\\data\\VOC\\ImageSets\\Main\\val.txt"

# 检查目录是否存在
if not os.path.exists(source_directory):
    print("指定的目录不存在")
    exit()

# 获取目录中的文件名（不包含后缀名）
file_names = [os.path.splitext(filename)[0] for filename in os.listdir(source_directory)]

# 创建并写入文件名到文本文件
with open(output_file_path, "w") as output_file:
    for file_name in file_names:
        output_file.write(file_name + "t\n")

print("文件名（不包含后缀名）已写入到", output_file_path)