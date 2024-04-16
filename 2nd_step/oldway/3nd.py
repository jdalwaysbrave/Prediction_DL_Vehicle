import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('output.csv')

# 检查每行第2，6，10列的值是否相等
for i, row in df.iterrows():
    cols = row.iloc[[1, 5, 9]].values
    if not np.all(cols == cols[0]):
        print(f"Row {i+1}: {cols}")

#没输出表示都相等