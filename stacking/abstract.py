# 导入需要的库
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 加载数据集和GRU、LSTM模型的预测结果
# 原数据
X_train, y_train = load_data()   # 加载训练集 
X_test, y_test = load_data()     # 加载测试集

gru_pred_train = load_gru_pred_train()  # GRU模型的训练集预测结果 输出
gru_pred_test = load_gru_pred_test()    # GRU模型的测试集预测结果 输出

lstm_pred_train = load_lstm_pred_train()  # LSTM模型的训练集预测结果 输出
lstm_pred_test = load_lstm_pred_test()    # LSTM模型的测试集预测结果 输出

# 将GRU和LSTM的预测结果作为特征融合在一起
X_train = np.column_stack((X_train, gru_pred_train, lstm_pred_train))
X_test = np.column_stack((X_test, gru_pred_test, lstm_pred_test))

# 使用随机森林模型拟合数据
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf.predict(X_test)

# 评估模型
evaluation(y_test, y_pred)