import os
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置自定义字体
font = FontProperties(fname='/home/chengyiqiu/code/INBA/resource/Fonts/Calibri.ttf')
label_size = 50
font_size = 20

# 数据文件路径
result_paths = {
    'BadNets': '/home/chengyiqiu/code/INBA/results/cifar10/badnet/fp/20250109160219/fine_pruning/plot_results.pth',
    'Blend': '/home/chengyiqiu/code/INBA/results/cifar10/blended/fp/20250109170059/fine_pruning/plot_results.pth',
    'FTrojan': '/home/chengyiqiu/code/INBA/results/cifar10/ftrojan/fp/20250109183958/fine_pruning/plot_results.pth',
    'WaNet': '/home/chengyiqiu/code/INBA/results/cifar10/wanet/fp/20250109182710/fine_pruning/plot_results.pth',
    'FIBA': '/home/chengyiqiu/code/INBA/results/cifar10/fiba/fp/20250109194638/fine_pruning/plot_results.pth',
    'CTRL': '/home/chengyiqiu/code/INBA/results/cifar10/ctrl/fp/20250109184050/fine_pruning/plot_results.pth',
    'DUBA': '/home/chengyiqiu/code/INBA/results/cifar10/duba/fp/20250109194723/fine_pruning/plot_results.pth',
    'Refool': '/home/chengyiqiu/code/INBA/results/cifar10/refool/fp/20250109173828/fine_pruning/plot_results.pth',
    'Phojan': '/home/chengyiqiu/code/INBA/results/cifar10/phase/fp/20250109193019/fine_pruning/plot_results.pth',
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

# 确定横轴（rounds）
rounds = list(range(1, length + 1))

# 超参数
sample_round = 25  # 每隔多少轮采样一次
sampled_rounds = rounds[::sample_round]

# 创建输出目录
output_dir = './FP'
os.makedirs(output_dir, exist_ok=True)

# 绘制并保存每种攻击的图
for name, metrics in data.items():
    acc_sampled = metrics['ACC'][::sample_round]
    asr_sampled = metrics['ASR'][::sample_round]
    
    plt.figure(figsize=(8, 6))
    plt.plot(sampled_rounds, acc_sampled, color='b', linestyle='-', label='BA', marker='o')
    plt.plot(sampled_rounds, asr_sampled, color='r', linestyle='--', label='ASR', marker='^')
    
    # 设置 X 轴和 Y 轴标签
    plt.xlabel('Pruned neurons', fontproperties=font, fontsize=label_size)
    plt.ylabel('BA/ASR (%)', fontproperties=font, fontsize=label_size)
    
    # 设置 Y 轴刻度字体大小
    plt.yticks(fontsize=font_size)
    plt.ylim(0, 110)
    
    # 添加图例
    leg = plt.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.15), ncol=2)

    # 直接设置图例中文字的字体大小
    for text in leg.get_texts():
        text.set_fontsize(fontsize=30)

    # 调整布局，确保图例不会与图表内容重叠
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 这里的rect参数调整了图表的边界，为图例留出空间
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{name}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

print(f'All sampled plots saved in {output_dir}')
