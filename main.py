import numpy as np
import matplotlib.pyplot as plt
from data import data
from indirect_adjustment import IndirectAdjustment


def get_fitting_parm(poi_data):
    """
    :param poi_data: 数据
    :return: 斜率平差值，截距平差值，中误差
    """
    # 用前两组数据计算参数近似值
    A = poi_data[:2].copy()
    A[:, 1] = 1
    b = poi_data[:2, 1]
    x0 = np.linalg.solve(A, b)
    a0 = x0[0]  # 斜率近似值
    b0 = x0[1]  # 截距近似值

    # 构造误差方程
    B = poi_data.copy()
    B[:, 1] = 1
    P = np.eye(B.shape[0])  # 认为数据等权，权阵为单位阵
    l = poi_data[:, 1] - poi_data[:, 0] * a0 - b0

    # 间接平差法计算改正数
    ia = IndirectAdjustment(B, P, l)
    a = a0 + ia.x[0]
    b = b0 + ia.x[1]
    return a, b, ia.sigma


def plot(a, b, poi_data):
    # 绘制数据点
    x_data = poi_data[:, 0]
    y_data = poi_data[:, 1]
    plt.scatter(x_data, y_data, color='blue', label='Data Points')

    # 绘制直线
    x_line = np.linspace(50, 100, 100)  # 横坐标范围从 50 到 100
    y_line = a * x_line + b  # 线性方程 y = a * x + b
    plt.plot(x_line, y_line, color='red', label=f'y = {a:.2f}x + {b:.2f}')
    plt.xlabel('2024')
    plt.ylabel('2025')
    plt.legend()
    plt.grid(True)
    plt.show()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    a, b, sigma = get_fitting_parm(data)
    print(f"中误差：{sigma}")
    plot(a, b, data)
