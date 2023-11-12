import numpy as np
import matplotlib.pyplot as plt

def ShowWave(data,a,b):
    values = data[0]

    # 只选择前一千个数据
    values_to_plot = values[a:b]

    # 创建 x 轴的数据，可以使用 np.arange 或者 np.linspace
    x = np.arange(len(values_to_plot))

    # 绘制图形
    plt.plot(x, values_to_plot, label='Data')

    # 添加标签和标题
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Plotting the First 1000 Values')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()
