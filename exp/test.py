import numpy as np
import matplotlib.pyplot as plt

# 初始化 (224, 224) 的ndarray
size = (224, 224)

# 创建一个渐变或者正态分布的 ndarray，X和Y是线性空间
X = np.linspace(-3, 3, size[0])
Y = np.linspace(-3, 3, size[1])
X, Y = np.meshgrid(X, Y)

# 使用正态分布公式来生成一个二维的分布图
Z = np.exp(-(X**2 + Y**2))

# 可视化
plt.imshow(Z, cmap='hot')
plt.colorbar()
plt.show()