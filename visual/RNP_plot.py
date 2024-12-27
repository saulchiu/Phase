import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置自定义字体
font = FontProperties(fname='/home/chengyiqiu/code/INBA/resource/Fonts/Calibri.ttf')

# 横轴：pruning ratio
x = [10, 30, 50, 70, 90]  # 固定的 pruning ratio

# 数据
BadNets_BA = [91.41, 89.23, 78.09, 18.06, 10.00]
BadNets_ASR = [43.66, 28.27, 38.95, 54.67, 90.88]

Blend_BA = [93.25, 91.87, 82.29, 20.49, 10.00]
Blend_ASR = [49.23, 35.08, 42.82, 7.12, 90.88]

WaNet_BA = [91.51, 87.85, 80.55, 31.99, 10.00]
WaNet_ASR = [16.79, 15.84, 14.94, 2.86, 0.87]

FTrojan_BA = [79.19, 63.51, 19.55, 10.02, 10.59]
FTrojan_ASR = [14.58, 12.53, 2.58, 1.08, 0.79]

FIBA_BA = [90.88, 89.25, 80.59, 43.27, 10.38]
FIBA_ASR = [17.60, 17.17, 16.37, 6.74, 1.15]

DUBA_BA = [48.00, 33.53, 11.50, 12.74, 9.59]
DUBA_ASR = [61.85, 52.37, 8.66, 40.95, 1.08]

Phojan_BA = [68.04, 56.64, 39.94, 18.43, 10.02]
Phojan_ASR = [89.36, 92.36, 89.71, 87.16, 90.07]

# 绘制图形
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
datasets = [
    (BadNets_BA, BadNets_ASR, 'BadNets'),
    (Blend_BA, Blend_ASR, 'Blend'),
    (WaNet_BA, WaNet_ASR, 'WaNet'),
    (FTrojan_BA, FTrojan_ASR, 'FTrojan'),
    (FIBA_BA, FIBA_ASR, 'FIBA'),
    (DUBA_BA, DUBA_ASR, 'DUBA'),
    (Phojan_BA, Phojan_ASR, 'Phojan')
]

# 自定义图例
legend_elements = []

# 循环绘制每组数据
for i, (BA, ASR, label) in enumerate(datasets):
    plt.plot(x, BA, color=colors[i], linestyle='-')
    plt.plot(x, ASR, color=colors[i], linestyle='--')
    plt.scatter(x, BA, color=colors[i], marker='o')  # 圆形表示 BA
    plt.scatter(x, ASR, color=colors[i], marker='^')  # 三角形表示 ASR
    # 添加自定义图例项
    legend_elements.append(plt.Line2D([0], [0], color=colors[i], marker='o', linestyle='-', label=f'{label} BA'))
    legend_elements.append(plt.Line2D([0], [0], color=colors[i], marker='^', linestyle='--', label=f'{label} ASR'))

# 设置 X 轴和 Y 轴标签
plt.xlabel('Pruning Ratio (%)', fontproperties=font)
plt.ylabel('Ratio (%)', fontproperties=font)

# 添加标题
plt.title('Pruning Ratio vs BA and ASR', fontproperties=font)

# 添加自定义图例
plt.legend(handles=legend_elements, loc='best', prop=font)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图像
# plt.savefig('RNP.pdf', format='pdf')
plt.savefig('RNP.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
# 显示图像
plt.show()
