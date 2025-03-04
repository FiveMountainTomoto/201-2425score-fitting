import numpy as np
import matplotlib.pyplot as plt
from data import data
from indirect_adjustment import IndirectAdjustment


def get_fitting_parm(poi_data, n):
    """
    :param n: 拟合曲线参数个数（曲线次数）
    :param poi_data: 数据
    :return: 斜率平差值，截距平差值，中误差
    """
    # 用前 n 组数据计算 n 个参数的近似值
    x = poi_data[:, 0]
    y0 = poi_data[:, 1]
    B0 = np.vander(x[:n], N=n, increasing=True)  # 使用 x 的前 n 个元素生成 N=n 列的矩阵
    parm0 = np.linalg.solve(B0, y0[:n])  # 参数近似值

    # 构造误差方程
    B = np.vander(x, N=n, increasing=True)  # 误差方程系数矩阵
    P = np.diag(poi_data[:, 2])  # 权阵
    l = y0 - B @ parm0

    # 间接平差法计算改正数
    ia = IndirectAdjustment(B, P, l)
    parm = parm0 + ia.x
    return parm, ia.sigma


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
    for i in range(2,10):
        parm, sigma = get_fitting_parm(data, i)
        print(sigma)
    # print(f"中误差：{sigma}")
    # plot(a, b, data)
