import pandas as pd
import numpy as np


np.set_printoptions(suppress=True)  # 关闭科学计数法

def create_dateset(trajectory, look_back=10):#lookback指用过去的10组数据来预测
    """
    :param trajectory:
    :param look_back:根据前10个点预测后1个点
    :return: dataY先由三维转成二维的
    """
    """dataX是一个三维的numpy数组，其形状为（n_samples，look_back，n_features），其中n_samples是数据集中的样本数，
    look_back是回顾过去的时间步数，n_features是每个数据点中的特征数。dataX包含将用于进行预测的输入序列。
    dataY是一个二维的numpy数组，其形状为（n_samples，look_back * n_features）。它包含了模型应该预测的输出序列。
    """
    dim = 10
    dataX, dataY = [], []
    for i in range(len(trajectory) - 2 * look_back - 1):
        a = trajectory[i: (i+look_back), :]
        dataX.append(a)
        b = trajectory[(i+look_back):(i + look_back+1), 1:3]#输出一行中的x，y坐标
        dataY.append(b)
    dataX = np.array(dataX, dtype='float64')
    dataY = np.array(dataY, dtype='float64')
    return dataX, dataY

def load_data(data_size):
    data_csv = pd.read_csv("./filtered_file.csv")#读取filtered file
    trajectory1 = np.array(data_csv, dtype=np.float64)  # trajectory1[:, 2:9] 原为2个数据 现为8个
    local_xy = trajectory1[0:data_size, 2:4]#data_size为选取的大小
    # 取前400个数据作为预测对象

    x = local_xy[:, 0]
    y = local_xy[:, 1]

#zscore_normalize(data):
    # meanx = np.mean(x)
    # stdx = np.std(x)
    # normalized_x = (x - meanx) / stdx
    # x=normalized_x

    # meany = np.mean(y)
    # stdy = np.std(y)
    # normalized_y = (y - meany) / stdy
    # y=normalized_y
    # print(x)
    # print(y)
# def minmax_normalize(data):
    min_x = np.min(x)
    max_x = np.max(x)
    normalized_x = (x - min_x) / (max_x - min_x)
    x= normalized_x+5

    min_y = np.min(y)
    max_y = np.max(y)
    normalized_y = (y - min_y) / (max_y - min_y)
    y= normalized_y+5
    print(min_x)
    print(max_x)
    print(min_y)
    print(max_y)
    print(1)
# def decimal_scaling_normalize(data):
#     max_val = np.max(data)
#     n = len(str(max_val))
#     normalized_data = data / (10 ** n)
#     return normalized_data

    # x归一化
    # print(np.max(x))
    # print(np.min(x))
    # scalar_x = np.max(x) - np.min(x)
    # x = ((x - np.min(x)) / scalar_x) + 0.001
    # print(x)
    # y归一化
    # scalar_y = np.max(y) - np.min(y)
    # y = (y - np.min(y)) / scalar_y
    # 速度信息归一化
    speed = trajectory1[0:data_size, 7]
    scalar_speed = np.max(speed) - np.min(speed)
    speed = (speed - np.min(speed)) / scalar_speed
    # 加速度信息归一化
    acc = trajectory1[0:data_size, 8]
    # scalar_acc = np.max(acc) - np.min(acc)
    # acc = (acc - np.min(acc)) / scalar_acc
    acc = (acc-np.min(acc))/(np.max(acc)-np.min(acc))*2-1
    # 转成列向量
    # print(x.shape)
    x = x.reshape(data_size, 1)
    y = np.array(y).reshape(data_size, 1)
    speed = np.array(speed).reshape(data_size, 1)
    acc = np.array(acc).reshape(data_size, 1)
    
    #构造一个序列号数组 seq，它的范围是从 0 到 data_size-1，将其形状从 (data_size,) 转换成 (data_size, 1)。
    seq = np.arange(data_size)
    scalar_seq = np.max(seq) - np.min(seq)
    seq = (seq - np.min(seq)) / scalar_seq
    seq = seq.reshape(data_size, 1)

    trajectory = np.hstack([seq, x, y,speed,acc])  # 5维数据， 1维是序号
    print(trajectory)  # 生成5维轨迹


    data_X, data_Y = create_dateset(trajectory)
    # 划分训练集和测试集，7/3
    train_size = int(len(data_X) * 0.7)
    # valsize=int(train_size1*0.2)
    # train_size=train_size1-valsize
    test_size = len(data_X) - train_size
   
    # x_val= data_X[train_size:train_size1]#验证集
    # y_val= data_X[train_size:train_size1]
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]#data_Y一列只有两维数据
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    # x_val= x_val.reshape(-1, 10, 5)
    # y_val = y_val.reshape(-1,2)
    train_X = train_X.reshape(-1, 10, 5)#保留第一维，10个时间步长和5个特征值
    train_Y = train_Y.reshape(-1,2)
    test_X = test_X.reshape(-1, 10, 5)
    test_Y = test_Y.reshape(-1,2)
    # 得到280个训练集， 120个测试集
    return train_X, train_Y, test_X,test_Y

load_data(832)