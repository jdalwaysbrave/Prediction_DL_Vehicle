import pandas as pd
import numpy as np
from keras.models import load_model
import importData_GRU as dt
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
""" 1.mse 等统一单位；时间连续性（不等间隔）; 学习率
    2.画出整体网络
    3.不同车的数据都拿来训练一下
"""""
if __name__ == '__main__':
    # predict
    data_size = 832
    train_x, train_y, test_x,test_y = dt.load_data(data_size)
    model = load_model("model_new/eps_200_bs_8_dp_64(256).h5")
    # test_y = model.predict(test_x)
    # test_y = test_y.reshape(-1, 1, 2)
    # print(test_x)
    
    # 评估模型
    # loss = model.evaluate(test_x, test_y, batch_size=8)
    # y_pred = model.predict(X_new)
    print(test_x)
    # 在测试集上进行预测
    y_pred = model.predict(test_x)
    print (test_x.shape)
    print (test_y.shape)
    # print(y_pred)
    print (y_pred.shape)
   
    # print(test_y)
    # 计算模型准确率
    # accuracy = accuracy_score(test_y, y_pred)
    # print('模型在测试集上的准确率为：', accuracy)
    # a=test_y[:,0]*test_y[:,0]+test_y[:,1]*test_y[:,1]
    # b=y_pred[:,0]*y_pred[:,0]+y_pred[:,1]*y_pred[:,1]
    # c=(b-a)/a
    # print(a)
    # print(b)
    # print(c)
    # test_x = test_x.reshape(-1, 5)
    # test_y = test_y.reshape(-1, 2)
    # save_x = pd.DataFrame(test_x, columns=['seq', 'Local_X', 'Local_Y','speed','acc'])
    # save_y = pd.DataFrame(test_y, columns=['Local_X', 'Local_Y'])
    # save_x.to_csv('test_x.csv')
    # save_y.to_csv('test_y.csv')


    # 计算相对平均绝对误差（rMAE）
    rmae = np.mean(np.abs((test_y - y_pred)/test_y))
    print('rMAE:', rmae)

    # # 计算均方误差（MSE）
    # mse = np.mean((test_y - y_pred)**2)
    # print('MSE:', mse)

    # # 计算均方根误差（RMSE）
    # rmse = np.sqrt(mse)
    # print('RMSE:', rmse)

    # # 计算决定系数（R^2）
    # ssr = np.sum((y_pred - np.mean(test_y))**2)
    # sst = np.sum((test_y - np.mean(test_y))**2)
    # r2 = 1 - ssr/sst
    # print('R^2:', r2)

    #还原预测坐标
    local_xy1 = y_pred[:, :]
    x1 = local_xy1[:, 0]
    y1 = local_xy1[:, 1]

    min_x=2230522.741
    max_x=6452679.267
    min_y=1375619.319
    max_y=1874268.248

    x1= (x1-5)*(max_x - min_x)+min_x
    y1= (y1-5)*(max_y - min_y)+min_y
    x1 = x1.reshape(-1, 1)
    y1= np.array(y1).reshape(-1, 1)
    # print(x1)
    # print(y1)
    xy = np.hstack([x1, y1])
    print(xy) 

    local_xy2 = test_y[:, :]
    x2 = local_xy2[:, 0]
    y2 = local_xy2[:, 1]
    x2= (x2-5)*(max_x - min_x)+min_x
    y2= (y2-5)*(max_y - min_y)+min_y
    x2 = x2.reshape(-1, 1)
    y2= np.array(y2).reshape(-1, 1)
    # print(x2)
    # print(y2)
    xy2 = np.hstack([x2, y2])
    print(xy2) 

#单位为ft， 1ft=30.48cm=0.3048m
#换算一下xy
    xy_m=xy*0.3048
    # print(xy_m)

    xy2_m=xy2*0.3048
    # print(xy2_m)




    # 合并两个矩阵
    merged_matrix = np.concatenate((xy_m, xy2_m), axis=1)

    print(merged_matrix) # 输出合并后矩阵的形状
    # 计算欧几里得距离
    # euclidean_dist = (cdist(xy, xy2))

    # # 计算曼哈顿距离
    # manhattan_dist = np.mean(cdist(actual_data, measured_data, 'cityblock'))

    # # 计算余弦相似度
    # cosine_sim = np.mean(np.dot(actual_data, measured_data.T) / (np.linalg.norm(actual_data, axis=1) * np.linalg.norm(measured_data, axis=1)))

    # # 计算相对误差
    # relative_error = np.mean(np.abs((measured_data - actual_data) / actual_data))

    # print("Euclidean Distance:", euclidean_dist)
    # print("Manhattan Distance:", manhattan_dist)
    # print("Cosine Similarity:", cosine_sim)
    # print("Relative Error:", relative_error)


 
    # plt.scatter(xy[:, 0], xy[:, 1], c='red', label='measured')
    # plt.scatter(xy2[:, 0], xy2[:, 1], c='blue', label='actual')
    # plt.legend()
    # plt.show()
