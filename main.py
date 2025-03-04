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


def plot_poi(poi_data):
    # 绘制数据点
    x_data = poi_data[:, 0]
    y_data = poi_data[:, 1]
    plt.scatter(x_data, y_data, color='blue', label='Data Points')


# 这个函数让KIMI写的，脏活给AI干，爽
def plot_curve(parm):
    """
    使用多项式系数绘制曲线
    :param parm: 多项式系数（按顺序为0,1,2...次系数）
    """
    # 横坐标范围
    x_line = np.linspace(40, 100, 100)

    # 多项式函数
    # y = c0 + c1*x + c2*x^2 + ... + cn*x^n
    # parm 是 [c0, c1, c2, ..., cn]
    # 因为 np.polyval 的输入是高次幂在前，低次幂在后，所以需要反转 parm
    poly = np.polyval(parm[::-1], x_line)

    # 动态生成方程字符串
    equation_parts = []
    for i, coeff in enumerate(parm):
        # 忽略系数为0的项
        if np.isclose(coeff, 0):
            continue
        # 指数部分
        power = i
        # 符号
        if coeff > 0:
            symbol = '+' if equation_parts else ''
        else:
            symbol = '-'
            coeff = abs(coeff)
        # 构建项
        if power == 0:
            part = f"{coeff:.2f}"
        elif power == 1:
            part = f"{coeff:.2f}x"
        else:
            part = f"{coeff:.2f}x^{power}"
        # 添加到方程字符串
        if symbol:
            equation_parts.append(f"{symbol} {part}")
        else:
            equation_parts.append(f"{part}")
    # 整合成方程字符串
    equation = "y = " + ' '.join(equation_parts)

    # 绘制曲线
    plt.plot(x_line, poly, color='red', label=equation)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # for i in range(2, 6):
    #     parm, sigma = get_fitting_parm(data, i)
    #     print(f"中误差：{sigma}")
    #     plot_curve(parm)
    parm, sigma = get_fitting_parm(data, 5)
    print(f"中误差：{sigma}")
    plot_curve(parm)
    plot_poi(data)

    plt.xlabel('2024')
    plt.ylabel('2025')
    plt.legend()
    plt.grid(True)

    plt.savefig('curve_plot.jpg', format='jpg', dpi=300)  # 保存拟合图片到当前文件夹
    plt.show()
