import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
import random
import numpy as np
# import seaborn as sns
import matplotlib
from scipy.interpolate import Rbf
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import queue
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD



# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')

# 搭建全连接神经网络回归
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        # 第一个隐含层
        self.hidden1 = nn.Linear(in_features=3, out_features=36, bias=True)
        # 第二个隐含层
        self.hidden2 = nn.Linear(36, 72)
        # 第三个隐含层
        self.hidden3 = nn.Linear(72, 144)
        # 回归预测层
        self.predict = nn.Linear(144, 3)

    # 定义网络前向传播路径
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        # 输出一个一维向量
        return output


def coordinate_xy():  # 生成传感单元的坐标
    x = []
    y = []
    rows = 10  # 行数
    columns = 16  # 列数
    interval_rows = 25  # 行间间距
    interval_columns = 25  # 行内间距
    a = 30  # a代表下边缘宽度
    b = 30  # b代表左边缘宽度

    for i in range(1, rows + 1):  # 有10行传感单元 行的顺序为从上到下 1-10
        for j in range(1, columns + 1):  # 每行16个传感单元 1-16
            if i % 2 == 1:
                xx = interval_columns * (j - 1) + b
            else:
                xx = interval_columns * (j - 1) + interval_columns / 2 + b
            yy = interval_rows * (rows - i) + a
            x.append(xx)
            y.append(yy)
    # x = [30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5, 142.5,
    #      167.5, 192.5, 217.5, 242.5, 267.5, 292.5, 317.5, 342.5, 367.5, 392.5, 417.5, 30, 55, 80, 105, 130, 155, 180,
    #      205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5, 142.5, 167.5, 192.5, 217.5, 242.5, 267.5,
    #      292.5, 317.5, 342.5, 367.5, 392.5, 417.5, 30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355,
    #      380, 405, 42.5, 67.5, 92.5, 117.5, 142.5, 167.5, 192.5, 217.5, 242.5, 267.5, 292.5, 317.5, 342.5, 367.5, 392.5,
    #      417.5, 30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5,
    #      142.5, 167.5, 192.5, 217.5, 242.5, 267.5, 292.5, 317.5, 342.5, 367.5, 392.5, 417.5, 30, 55, 80, 105, 130, 155,
    #      180, 205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5, 142.5, 167.5, 192.5, 217.5, 242.5,
    #      267.5, 292.5, 317.5, 342.5, 367.5, 392.5, 417.5]
    # y = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 230, 230, 230, 230, 230, 230,
    #      230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
    #      205, 205, 205, 205, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 155, 155,
    #      155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 130, 130, 130, 130, 130, 130, 130, 130,
    #      130, 130, 130, 130, 130, 130, 130, 130, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
    #      105, 105, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 55, 55, 55, 55, 55, 55, 55, 55, 55,
    #      55, 55, 55, 55, 55, 55, 55, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    return x, y
    pass


def read_txt():
    # 这里为读取在手术室里采集的磁场数据
    file_path = r'C:\Users\MACBOOKAIR\Desktop\BYJOB\data\SensorArray_ReadData\OperatingTable0610\liuzijietou.txt'
    with open(file_path, 'r') as file:
        content = file.read()

    # 分割字符串
    rows = content.split('\n')

    # 创建二维矩阵
    matrix = []
    for row in rows:
        if "from" in row:
            continue  # 跳过包含特定字符串的行
        # 对每个子字符串按制表符分割
        columns = row.split('\t')
        # 检查分割后的列数是否为484
        if len(columns) == 484:
            # 将分割后的结果添加到矩阵中
            matrix.append(columns)
    # raw_data = np.array(matrix)
    # # 排除最后一列数据（时间戳）
    # data = raw_data[:, :-4].astype(int)  # 读取txt -1,带IMU的txt -4
    data = np.array(matrix)[:, :-4].astype(int)
    # 减去初始值
    data -= data[0, :]
    return data
    pass


def get_F(dataB):
    model_array = []
    model_indices = []
    dataP = dataB.astype(float)
    for iic in range(1, 4):
        for Num in range(1, 5):
            if ((iic == 1 or iic == 3) and Num == 4): continue
            for Node in range(1, 17):
                if (iic == 1 and Num == 1) or (iic == 1 and Num == 2) or (iic == 2 and Num == 1) or (iic == 2 and Num == 2) or (iic == 3 and Num == 1):
                    if 12 <= Node <= 16:
                        model = torch.load(f'C:\\Users\\MACBOOKAIR\\Desktop\\BYJOB\\dynamic_display\\dnnCludeMini\\dnn-{iic}-{Num}-{Node}Mini-10000.pt', map_location=torch.device('cpu'))
                        model_array.append(model)
                        model_indices.append(len(model_array) - 1)
                    else:
                        model = torch.load(f'C:\\Users\\MACBOOKAIR\\Desktop\\BYJOB\\dynamic_display\\dnnCludeMini\\dnn-{iic}-{Num}-{Node}-10000.pt',map_location=torch.device('cpu'))
                        model_array.append(model)
                elif (iic == 1 and Num == 3) or (iic == 2 and Num == 3) or (iic == 2 and Num == 4) or (iic == 3 and Num == 2) or (iic == 3 and Num == 3):
                    model = torch.load(f'C:\\Users\\MACBOOKAIR\\Desktop\\BYJOB\\dynamic_display\\dnnCludeMini\\dnn-{iic}-{Num}-{Node}Mini-10000.pt', map_location=torch.device('cpu'))
                    model_array.append(model)
                    model_indices.append(len(model_array) - 1)
                else :
                    model = torch.load(f'C:\\Users\\MACBOOKAIR\\Desktop\\BYJOB\\dynamic_display\\dnnCludeMini\\dnn-{iic}-{Num}-{Node}-10000.pt',map_location=torch.device('cpu'))
                    model_array.append(model)

    for i in range(dataB.shape[0]):
        row_data = dataB[i]
        for j in range(0, dataB.shape[1], 3):
            # 提取当前行中的每三个列数据
            data = row_data[j:j + 3]
            data = data.astype(float)
            data /= 1000  # 标准化
            #     # 将输入数据转换为 PyTorch 的 Tensor 类型，并显式地指定数据类型为 float
            input_tensor = torch.tensor(data, dtype=torch.float)
            # 按照指定顺序调用模型
            model_index = j // 3 % len(model_indices)
            model = model_array[model_index]
            # 进行模型预测，并将结果转换为 NumPy 数组
            y_pred = model(input_tensor).detach().numpy()
            # print(y_pred)
            # if (y_pred[2] < 0):
            if any(model_index == i for index in model_indices):
                dataP[i, j:j + 3] = y_pred/132.0*1000.0
            else:
                dataP[i, j:j + 3] = y_pred/40.0*1000.0

            # else:dataP[i, j:j + 3] = y_pred
            print(dataP[i, j:j + 3])
    return dataP


def get_B(dataB, XYZ, ti):
    if XYZ in [0, 1, 2]:
        # XYZ取值范围为0, 1, 2
        # 提取BX/BY/BZ列
        BXYZ = dataB[:, XYZ::3]
        # 提取time时刻的数据
        BXYZ_time = BXYZ[ti, :].tolist()
        return BXYZ_time
    else:
        raise ValueError("Invalid value for XYZ. XYZ must be 0, 1, or 2.")


def cloud(x, y, z):
    # 单次显示
    # 定义目标网格
    num = 1000  # 网格点数量
    xi = np.linspace(min(x), max(x), num)
    yi = np.linspace(min(y), max(y), num)
    xi, yi = np.meshgrid(xi, yi)

    # 使用griddata函数进行插值
    # zi = griddata((x, y), z, (xi, yi), method='cubic')  # 'linear'线性插值、'nearest'最近邻插值和'cubic'三次样条插值

    # 使用Rdf径向基函数进行插值
    # 创建 Rbf 插值对象
    # rbf = Rbf(x, y, z, function='quadratic')
    rbf = Rbf(x, y, z, function='gaussian')  # multiquadric多重inverse逆gaussian高斯linear线性cubic三次thin_plate薄板
    # 使用 Rbf 进行插值
    zi = rbf(xi, yi)

    # 绘制云图
    plt.contourf(xi, yi, zi, cmap='rainbow')
    plt.colorbar()  # 添加颜色条
    plt.scatter(x, y, s=1, c=z, cmap='rainbow', edgecolors='black')  # 绘制数据点
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cloud Map')
    plt.show()
    pass


def main():
    dataB = read_txt()
    dataB = np.array(dataB)

    print(dataB)
    print(dataB.shape)

    dataP = get_F(dataB)
    array_size = np.shape(dataP)

    print(array_size)


    x, y = coordinate_xy()

    # 单次绘图
    # z = get_B(XYZ=0, ti=788)  # X0 Y1 Z2
    # cloud(x,y,z)

    # # 单个动态绘图
    # fig, ax = plt.subplots()
    # t = queue.Queue()
    # for i in range(100, 600):
    #     t.put(i)
    #     # 定义目标网格
    # num = 80  # 网格点数量
    # xi = np.linspace(min(x), max(x), num)
    # yi = np.linspace(min(y), max(y), num)
    # xi, yi = np.meshgrid(xi, yi)
    # cbar = None
    #
    # def update(frame):
    #     nonlocal cbar
    #     ax.clear()
    #     # 更新z
    #     ti = t.get()
    #     print(ti)
    #     z = get_B(dataB=dataB, XYZ=1, ti=ti)
    #
    #     # 创建 Rbf 插值对象
    #     rbf = Rbf(x, y, z, function='gaussian')
    #     # 使用 Rbf 进行插值
    #     zi = rbf(xi, yi)
    #     # 绘制云图
    #     # ax.contourf(xi, yi, zi, levels=50, cmap='rainbow')  # lecels颜色梯度等级 camp颜色映射
    #     cax = ax.contourf(xi, yi, zi, levels=10, cmap='rainbow')
    #     # 删除旧的颜色条，如果存在的话
    #     if cbar is not None:
    #         cbar.remove()
    #     # 添加颜色条
    #     cbar = plt.colorbar(cax)
    #     ax.scatter(x, y, s=1, c=z, cmap='rainbow', edgecolors='black')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_title('Cloud Map')
    #     print("time:", time.time())
    #
    # # 创建 FuncAnimation 对象
    # ani = FuncAnimation(fig, update, frames=range(400), interval=1)  # frames帧数 interval帧间的间隔时间
    # plt.show()

    # 三个动态绘图
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # 创建队列t 用于图像更新时,z中ti的递增
    t = queue.Queue()
    for i in range(1, 900):
        t.put(i)

    # 定义目标网格
    num = 100  # 网格点数量
    xi = np.linspace(min(x), max(x), num)
    yi = np.linspace(min(y), max(y), num)
    xi, yi = np.meshgrid(xi, yi)

    cbars = [None, None, None]  # 存储颜色条对象的列表

    def update(frame):
        nonlocal cbars
        ti = t.get()
        print(ti)
        for i, ax in enumerate(axes):
            ax.clear()
            # 更新z
            z = get_B(dataB=dataP, XYZ=i, ti=ti)  # 获取不同图形的z值
            # 创建 Rbf 插值对象
            rbf = Rbf(x, y, z, function='gaussian')
            # 使用 Rbf 进行插值
            zi = rbf(xi, yi)
            # 绘制云图
            cax = ax.contourf(xi, yi, zi, levels=100, cmap='rainbow')
            # 删除旧的颜色条，如果存在的话
            if cbars[i] is not None:
                cbars[i].remove()
            # 添加颜色条
            cbars[i] = plt.colorbar(cax, ax=ax)
            ax.scatter(x, y, s=1, c=z, cmap='rainbow', edgecolors='black')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Cloud Map')
        print("time:", time.time())

    # 创建 FuncAnimation 对象
    ani = FuncAnimation(fig, update, frames=range(400), interval=1)  # frames帧数 interval帧间的间隔时间
    plt.show()

    pass


if __name__ == '__main__':
    main()

    # x= [30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5, 142.5,
    #      167.5, 192.5, 217.5, 242.5, 267.5, 292.5, 317.5, 342.5, 367.5, 392.5, 417.5, 30, 55, 80, 105, 130, 155, 180,
    #      205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5, 142.5, 167.5, 192.5, 217.5, 242.5, 267.5,
    #      292.5, 317.5, 342.5, 367.5, 392.5, 417.5, 30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355,
    #      380, 405, 42.5, 67.5, 92.5, 117.5, 142.5, 167.5, 192.5, 217.5, 242.5, 267.5, 292.5, 317.5, 342.5, 367.5, 392.5,
    #      417.5, 30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5,
    #      142.5, 167.5, 192.5, 217.5, 242.5, 267.5, 292.5, 317.5, 342.5, 367.5, 392.5, 417.5, 30, 55, 80, 105, 130, 155,
    #      180, 205, 230, 255, 280, 305, 330, 355, 380, 405, 42.5, 67.5, 92.5, 117.5, 142.5, 167.5, 192.5, 217.5, 242.5,
    #      267.5, 292.5, 317.5, 342.5, 367.5, 392.5, 417.5]
    # y= [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 230, 230, 230, 230, 230, 230,
    #      230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
    #      205, 205, 205, 205, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 155, 155,
    #      155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 130, 130, 130, 130, 130, 130, 130, 130,
    #      130, 130, 130, 130, 130, 130, 130, 130, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
    #      105, 105, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 55, 55, 55, 55, 55, 55, 55, 55, 55,
    #      55, 55, 55, 55, 55, 55, 55, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    # z= [48, 16, -77, -180, 77, -72, 0, -88, -43, 130, 68, -124, -148, 43, 40, -11, 49, -151, 20, 37, 28, 32, -308, 337,
    #      121, -35, -139, -60, -97, 125, 77, -34, -21, -37, 58, 105, 3, 14, -102, 35, -13, 148, 71, -108, -112, 12, 93,
    #      44, -72, -73, -19, 36, 57, -84, -120, 77, 49, 65, 125, 38, 37, -11, 132, 57, -19, -38, -57, -8, 48, 3, -30, 87,
    #      -23, 87, 95, 16, -46, 2, 97, 90, -170, -209, -39, 16, 49, -15, 42, -50, 26, 68, 47, 7, -47, 76, 19, 18, -78,
    #      -118, -133, -32, -11, 27, 10, 183, -100, 24, 87, 38, -56, 8, 80, 135, -107, -132, -97, -31, -5, -21, 154, -11,
    #      -45, 80, 33, -24, -105, -82, 130, 88, -94, -157, -143, -63, 41, 173, 140, 320, -220, 13, 8, -30, -80, -54, 100,
    #      139, -157, -217, -169, 2, 95, 236, 260, -46, -145, -113, -164, -173, -132, -44, 94, 54]
