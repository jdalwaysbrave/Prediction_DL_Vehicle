import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
import importData_GRU as dt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Lambda, Dense

def train_model(train_x, train_y, epochs, batch_size, hidden_size):
    """
    :param train_x:
    :param train_y: LSTM训练所需的训练集
    :return: 训练得到的模型
    """
#如果 hidden_size 设得太小，模型可能无法很好地捕捉数据中的模式和规律，导致欠拟合；如果 hidden_size 设得太大，则可能会导致模型过拟合，泛化能力下降。
#一般来说，可以从小到大逐步尝试不同的 hidden_size 值，观察训练集和验证集的损失值的变化情况，选取一个使得损失值最小的值作为最终的 hidden_size
    # 定义GRU网络
    
    model = Sequential()
    model.add(GRU(hidden_size, input_shape=(10,5), return_sequences=True))
    model.add(Lambda(lambda x: x[:, -1, :]))
    model.add(Dense(2))

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
    model.summary()
   
    return model



if __name__ == '__main__':
    """
    LSTM输入三维的数据，(seq, local_x, local_y)
    根据前10步预测当前时刻后10步的轨迹
    """
    data_size = 832
    train_x, train_y, test_x,test_y = dt.load_data(data_size)#出问题了
    # train_x.shape(280, 10, 2)
    print(train_y.shape)
    print(train_x.shape)
    # print(train_x)
    # print(train_y)
    epochs = 200
    batch_size = 8      
    hidden_size=64  #较小的 hidden_size，比如 64、128 或 256 等。而对于一些较为复杂的任务，比如机器翻译、语音识别等，可能需要使用更大的 hidden_size，比如 512、1024 或更大的值
    model = train_model(train_x, train_y, epochs, batch_size, hidden_size)
    model_name = "./model_new/eps_" + str(epochs) + "_bs_" + str(batch_size) + "_dp_" + str(hidden_size) + "(256).h5"
    model.save(model_name)
