import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)+ np.cos(x)+ np.random.random(100)

y_filtered= savgol_filter(y,99,3)

fig=plt.figure()
ax = fig.subplots()
p = ax.plot(x,y,'-*')
p = ax.plot(x,y_filtered,'g')
plt.show()
