import os
import xml.etree.ElementTree as ET

# 定义替换规则，键为旧的<name>标签内容，值为新的<name>标签内容
replacements = {
    'huxi': 'nonoviposition',
    'tuisha': 'nonoviposition',
    'tuisa': 'nonoviposition',
    'chanluan': 'oviposition',
    'huluan': 'oviposition',
    # 添加更多替换规则
}

# 指定包含XML文件的文件夹路径
folder_path = "D:\\Projects\\ML\\data\\VOC\\temp"

# 遍历文件夹中的所有XML文件
for filename in os.listdir(folder_path):
    if filename.endswith(".xml"):
        xml_file_path = os.path.join(folder_path, filename)

        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 遍历XML文件中的<name>标签
        for name_element in root.iter("name"):
            # 获取<name>标签的当前内容
            current_name = name_element.text

            # 检查是否有替换规则，并进行替换
            if current_name in replacements:
                name_element.text = replacements[current_name]

        # 保存修改后的XML文件
        tree.write(xml_file_path)

print("处理完成。")
