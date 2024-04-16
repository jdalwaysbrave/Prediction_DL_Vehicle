# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('result1.csv', header=None)

# # 按第二列的值进行分类，并删除不符合条件的行
# df_grouped = df.groupby(1).filter(lambda x: len(x) == 3)

# # 保存处理后的CSV文件
# df_grouped.to_csv('processed_data.csv', index=False)
import pandas as pd
import numpy as np
from math import degrees, acos
# 读取CSV文件
df = pd.read_csv('processed_data.csv')
print(df.shape)
# 将每三行分为一组，并提取坐标数据
coords = df.values.reshape(-1, 3, 4)
print(coords.shape)
agg=coords.reshape(-1,12)
print(agg)
np.savetxt('output.csv', agg, delimiter=',', fmt='%.6f')
# # print(coords)
# # 计算每组数据构成的三角形的角度，并保存到DataFrame中
# angles = []
# for i, group in enumerate(coords):
#     x1, y1 = group[0]
#     x2, y2 = group[1]
#     x3, y3 = group[2]
#     a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
#     c = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
#     cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
#     cos_beta = (a**2 + c**2 - b**2) / (2 * a * c)
#     cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
#     alpha = degrees(np.arccos(np.clip(cos_alpha, -1, 1)))
#     beta = degrees(np.arccos(np.clip(cos_beta, -1, 1)))
#     gamma = degrees(np.arccos(np.clip(cos_gamma, -1, 1)))
#     angles.append([i+1, x1, y1, x2, y2, x3, y3, alpha, beta, gamma])

# df_angles = pd.DataFrame(angles, columns=['group', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'alpha', 'beta', 'gamma'])
# df_angles.to_csv('angles.csv', index=False)