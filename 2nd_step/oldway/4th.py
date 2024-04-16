import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('output.csv')

# 计算每行三辆车构成的三角形的三个角的角度
angles = []
for i, row in df.iterrows():
    x1, y1 = row[2], row[3]
    x2, y2 = row[6], row[7]
    x3, y3 = row[10], row[11]
    a = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    b = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
    c = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    cosA = (b**2 + c**2 - a**2) / (2 * b * c)
    cosB = (a**2 + c**2 - b**2) / (2 * a * c)
    cosC = (a**2 + b**2 - c**2) / (2 * a * b)
    angleA = np.arccos(cosA) * 180 / np.pi
    angleB = np.arccos(cosB) * 180 / np.pi
    angleC = np.arccos(cosC) * 180 / np.pi
    angles.append([angleA, angleB, angleC])

# 将计算结果转换为一个新的DataFrame对象，并与原始数据合并
angles_df = pd.DataFrame(angles, columns=['angleA', 'angleB', 'angleC'])
df = pd.concat([df, angles_df], axis=1)

# 将计算结果输出到CSV文件
df.to_csv('output2.csv', index=False)