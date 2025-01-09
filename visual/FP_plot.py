import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置自定义字体
font = FontProperties(fname='/home/chengyiqiu/code/INBA/resource/Fonts/Calibri.ttf')

# 数据文件路径
result_paths = {
    'BadNets': '/home/chengyiqiu/code/INBA/results/cifar10/badnet/fp/20250109160219/fine_pruning/plot_results.pth',
    # 'Blend': '/home/chengyiqiu/code/INBA/results/cifar10/blended/rnp/20241226124509/IBAU/plot_results.pth',
    # 'FTrojan': '/home/chengyiqiu/code/INBA/results/cifar10/ftrojan/rnp/20241226124550/IBAU/plot_results.pth',
    # 'WaNet': '/home/chengyiqiu/code/INBA/results/cifar10/wanet/rnp/20241226124532/IBAU/plot_results.pth',
    # 'FIBA': '/home/chengyiqiu/code/INBA/results/cifar10/fiba/rnp/20241226135918/IBAU/plot_results.pth',
    # 'CTRL': '/home/chengyiqiu/code/INBA/results/cifar10/ctrl/rnp/20250104124752/IBAU/plot_results.pth',
    # 'DUBA': '/home/chengyiqiu/code/INBA/results/cifar10/duba/rnp/20241226135946/IBAU/plot_results.pth',
    # 'Refool': '/home/chengyiqiu/code/INBA/results/cifar10/refool/rnp/20250104124818/IBAU/plot_results.pth',
    # 'Phojan': '/home/chengyiqiu/code/INBA/results/cifar10/phase/rnp/20241224201655/IBAU/plot_results.pth',
}

# 从文件中读取数据
data = {}
length = None
for name, path in result_paths.items():
    res = torch.load(path)
    acc_list = res['acc_list']
    asr_list = res['asr_list']
    # 断言所有列表长度一致
    if length is None:
        length = len(acc_list)
    assert length == len(acc_list), "Length Error"
    assert length == len(asr_list), "Length Error"
    data[name] = {'ACC': acc_list, 'ASR': asr_list}

# 超参数
sample_round = 10  # 每隔多少轮采样一次

# 确定横轴（rounds）
rounds = list(range(1, length + 1))
sampled_rounds = rounds[::sample_round]

# 设置绘图参数
plt.figure(figsize=(12, 8))
colors = {
    'BadNets': 'b',  # 蓝色
    # 'Blend': 'g',    # 绿色
    # 'FTrojan': 'r',  # 红色
    # 'WaNet': 'c',    # 青色
    # 'FIBA': 'm',     # 品红色
    # 'CTRL': 'y',     # 黄色
    # 'DUBA': 'orange',# 橙色
    # 'Refool': 'purple', # 紫色
    # 'Phojan': 'k',   # 黑色
}

# 绘制曲线
for name, metrics in data.items():
    acc_sampled = metrics['ACC'][::sample_round]
    asr_sampled = metrics['ASR'][::sample_round]
    plt.plot(sampled_rounds, acc_sampled, color=colors[name], linestyle='-', label=f'{name} ACC', marker='o')
    plt.plot(sampled_rounds, asr_sampled, color=colors[name], linestyle='--', label=f'{name} ASR', marker='^')

# 设置 X 轴和 Y 轴标签
plt.xlabel('Filters Pruned', fontproperties=font)
plt.ylabel('Values (%)', fontproperties=font)

# 添加标题
plt.title(f'Filters Pruned vs ACC and ASR (Sampled every {sample_round} rounds)', fontproperties=font)

# 添加图例
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop=font)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 保存图像
plt.savefig('FP_ratio.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# 显示图像
plt.show()
