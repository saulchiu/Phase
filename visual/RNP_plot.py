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
    'BadNets': '/home/chengyiqiu/code/INBA/results/cifar10/badnet/rnp/20241226124414/rnp_plot/plot_results.pth',
    'Blend': '/home/chengyiqiu/code/INBA/results/cifar10/blended/rnp/20241226124509/rnp_plot/plot_results.pth',
    'Refool': '/home/chengyiqiu/code/INBA/results/cifar10/refool/rnp/20250104124818/rnp_plot/plot_results.pth',
    'WaNet': '/home/chengyiqiu/code/INBA/results/cifar10/wanet/rnp/20241226124532/rnp_plot/plot_results.pth',
    'FTrojan': '/home/chengyiqiu/code/INBA/results/cifar10/ftrojan/rnp/20241226124550/rnp_plot/plot_results.pth',
    'CTRL': '/home/chengyiqiu/code/INBA/results/cifar10/ctrl/rnp/20250104124752/rnp_plot/plot_results.pth',
    'FIBA': '/home/chengyiqiu/code/INBA/results/cifar10/fiba/rnp/20241226135918/rnp_plot/plot_results.pth',
    'DUBA': '/home/chengyiqiu/code/INBA/results/cifar10/duba/rnp/20241226135946/rnp_plot/plot_results.pth',
    'Phojan': '/home/chengyiqiu/code/INBA/results/cifar10/phase/rnp/20241224201655/rnp_plot/plot_results.pth',
}

# 从文件中读取数据
data = {}
length = None
for name, path in result_paths.items():
    res = torch.load(path)
    acc_list = [x * 100 for x in res['acc_list']]
    asr_list = [x * 100 for x in res['asr_list']]
    # 断言所有列表长度一致
    if length is None:
        length = len(acc_list)
    assert length == len(acc_list), "Length Error"
    assert length == len(asr_list), "Length Error"
    data[name] = {'ACC': acc_list, 'ASR': asr_list}

# 确定横轴（pruning ratio）
pruning_ratios = [i * 0.1 for i in range(1, 11)]  # 0.1 到 1.0

# 创建保存图像的目录
output_dir = './RNP'
os.makedirs(output_dir, exist_ok=True)

# 绘制每种攻击的曲线图
for name, metrics in data.items():
    plt.figure(figsize=(8, 6))
    acc = metrics['ACC']
    asr = metrics['ASR']
    plt.plot(pruning_ratios, acc, linestyle='-', label=f'BA', marker='o', color='b')
    plt.plot(pruning_ratios, asr, linestyle='--', label=f'ASR', marker='^', color='r')
    
    # 设置 X 轴和 Y 轴标签
    plt.xlabel('Pruning ratio', fontproperties=font, fontsize=label_size)
    plt.ylabel('BA/ASR (%)', fontproperties=font, fontsize=label_size)
    
    # 设置 X 轴刻度范围
    plt.xticks(pruning_ratios, fontsize=font_size)  # 设置 X 轴刻度及字体大小
    
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
    
    # 保存图像
    save_path = os.path.join(output_dir, f'{name}.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    
    # 显示图像（可选）
    plt.close()

print(f"All plots saved in {output_dir}")
