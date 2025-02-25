import math
import numpy as np


class IndirectAdjustment:
    B: np.ndarray
    P: np.ndarray
    l: np.ndarray
    V: np.ndarray
    x: np.ndarray
    sigma: float

    def __init__(self, B: np.ndarray, P: np.ndarray, l: np.ndarray):
        """
        :param B: 误差方程系数矩阵
        :param P: 权阵
        :param l: 误差方程常数向量
        """
        self.B = B
        self.P = P
        self.l = l
        rowCount, colCount = B.shape
        self.freedom = rowCount - colCount  # 计算自由度
        self.__calculate()

    def __calculate(self):
        B = self.B
        P = self.P
        l = self.l
        Nbb = B.T @ P @ B  # 法方程系数矩阵
        Nbbi = np.linalg.inv(Nbb)
        x = self.x = Nbbi @ B.T @ P @ l  # 参数改正数
        V = self.V = B @ x - l  # 观测值改正数
        self.sigma = math.sqrt(V.T @ P @ V / self.freedom)  # 单位权中误差
