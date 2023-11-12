import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(src_folder, dest_folder):
    for xml_file in os.listdir(src_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(src_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像尺寸信息
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # 处理目标对象
            yolo_data = []
            for obj in root.findall('object'):
                label = obj.find('name').text

                # 假设您已经有一个将类别名映射到整数的字典
                class_mapping = {'huxi': 0, 'tuisha': 0, 'tuisa': 0, 'huluan': 1, 'chanluan':1}
                class_id = class_mapping[label]

                # 获取边界框信息
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # 计算YOLO格式的边界框坐标
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height

                yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

            # 保存为.txt文件
            txt_file = os.path.splitext(xml_file)[0] + '.txt'
            txt_path = os.path.join(dest_folder, txt_file)
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_data))

if __name__ == "__main__":
    src_folder = 'D:\\Projects\\ML\\data\\dani\\train\\xmllabels'
    dest_folder = 'D:\\Projects\\ML\\data\\dani2class\\train\\label'

    # 创建目标文件夹
    os.makedirs(dest_folder, exist_ok=True)

    # 转换.xml文件为YOLOv5的.txt文件
    convert_xml_to_yolo(src_folder, dest_folder)

    print("XML文件已成功转换为YOLOv5的.txt文件！")
