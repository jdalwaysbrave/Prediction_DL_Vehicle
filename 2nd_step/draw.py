import matplotlib.pyplot as plt

# 预测位置数据
predicted = [[679857.06, 419314.97],
             [679857.94, 419332.56],
             [679859.44, 419305.3],
             [679859.56, 419339.47],
             [679860.75, 419356.9]]

# 实际位置数据
actual = [[679857.2739, 419317.2236],
          [679858.8576, 419333.4677],
          [679859.7071, 419307.1674],
          [679859.6912, 419340.4366],
          [679860.8342, 419357.7712]]

# 绘制预测位置和实际位置的散点图
fig, ax = plt.subplots()
ax.scatter([x[0] for x in predicted], [x[1] for x in predicted], label='Predicted')
ax.scatter([x[0] for x in actual], [x[1] for x in actual], label='Actual')
ax.legend()
plt.show()

import math

# 预测位置数据
predicted = [[679857.06, 419314.97],
             [679857.94, 419332.56],
             [679859.44, 419305.3],
             [679859.56, 419339.47],
             [679860.75, 419356.9]]

# 实际位置数据
actual = [[679857.2739, 419317.2236],
          [679858.8576, 419333.4677],
          [679859.7071, 419307.1674],
          [679859.6912, 419340.4366],
          [679860.8342, 419357.7712]]

# 计算各自的欧氏距离
distances = []
for i in range(len(predicted)):
    dist = math.sqrt((predicted[i][0] - actual[i][0])**2 + (predicted[i][1] - actual[i][1])**2)
    distances.append(dist)
    print(f"车辆{i+1}的欧氏距离为：{dist:.2f}")