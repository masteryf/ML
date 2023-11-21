import numpy as np
from pyts.image import MarkovTransitionField
import pandas as pd
from ast import literal_eval
import cv2
from utils.showWave import ShowWave

mtf = MarkovTransitionField()


# # 从 CSV 文件中读取数据
# data = pd.read_csv("D:\Projects\ML\dataset\V5.csv")
# data = data["Tag"][120]
# data = literal_eval(data)
#
# print(len(data))
# ShowWave(data)
# # 创建 Markov Transition Field 对象
#
#
# # 将数据转换为图像
# x = np.array(data).reshape(1, -1)
# image = mtf.transform(x)
# print(image.shape)
# # 获取图像的高度和宽度
# height, width = image.shape[1], image.shape[2]
#
# # 将图像转换为灰度图像
# gray_image = image.reshape(height, width)
#
# # 显示灰度图像
# cv2.imshow('1', gray_image)
#
# # 等待用户按下任意按键后关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def MarkovTrans(data):
    x = np.array(data).reshape(1, -1)
    image = mtf.transform(x)
    height, width = image.shape[1], image.shape[2]
    gray_image = image.reshape(height, width)
    return gray_image
