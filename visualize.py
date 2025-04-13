import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_network_parameters(model_path):
    """可视化神经网络参数"""
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 获取参数
    W1 = model.params['W1']  # (3072, hidden_size)
    W2 = model.params['W2']  # (hidden_size, 10)
    hidden_size = W1.shape[1]

    # 创建主画布（增大图形尺寸并调整DPI）
    fig = plt.figure(figsize=(20, 24), dpi=100, layout='constrained')
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[1.2, 1.2, 2],  # 增加高度比例
                  width_ratios=[1, 1],
                  hspace=0.4, wspace=0.3)  # 增加子图间距

    # --------------------------
    # 权重分布直方图
    # --------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(W1.flatten(), bins=100, alpha=0.7,
             label=f'W1 (Input→Hidden, {W1.size} weights)',
             color='blue')
    ax1.hist(W2.flatten(), bins=100, alpha=0.7,
             label=f'W2 (Hidden→Output, {W2.size} weights)',
             color='orange')
    ax1.set_title('Weight Value Distribution', pad=15, fontsize=12)
    ax1.set_xlabel('Weight Value', fontsize=10)
    ax1.set_ylabel('Frequency (log)', fontsize=10)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.tick_params(axis='both', which='major', labelsize=9)

    # --------------------------
    # 输入层权重可视化 (展示部分)
    # --------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # 创建更宽松的内部网格
    inner_gs = gs[0, 1].subgridspec(4, 4, wspace=0.3, hspace=0.5)

    # 随机选择16个隐藏神经元
    sample_neurons = np.random.choice(hidden_size, 16, replace=False)
    sample_weights = W1[:, sample_neurons].reshape(32, 32, 3, 16).transpose(3, 0, 1, 2)

    for i in range(16):
        ax = fig.add_subplot(inner_gs[i])
        neuron_weights = sample_weights[i]
        neuron_weights = (neuron_weights - neuron_weights.min()) / (neuron_weights.max() - neuron_weights.min() + 1e-8)
        ax.imshow(neuron_weights)
        ax.set_title(f'Neuron {sample_neurons[i]}', fontsize=8, pad=1)
        ax.axis('off')

    fig.suptitle('Input Layer Weight Patterns (Random 16 Neurons)',
                 y=0.97, fontsize=12)

    # --------------------------
    # 输出层权重热力图
    # --------------------------
    ax3 = fig.add_subplot(gs[1:, :])
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 动态调整显示神经元数量
    display_neurons = min(500, hidden_size)  # 最多显示300个神经元
    heatmap_data = W2.T[:, :display_neurons]

    im = ax3.imshow(heatmap_data, cmap='coolwarm', aspect='auto',
                    extent=[0, display_neurons, 9.5, -0.5],
                    interpolation='nearest')

    ax3.set_title(f'Hidden-to-Output Weight Matrix (First {display_neurons}/{hidden_size} Neurons)',
                  pad=15, fontsize=12)
    ax3.set_xlabel('Hidden Neuron Index', fontsize=10)
    ax3.set_ylabel('Output Class', fontsize=10)
    ax3.set_yticks(np.arange(10))
    ax3.set_yticklabels(class_names, fontsize=9)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.grid(False)

    # 添加颜色条（调整位置和大小）
    cbar = fig.colorbar(im, ax=ax3, pad=0.03, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Weight Value', rotation=270, labelpad=15, fontsize=10)

    # 保存图形（调整边界）
    plt.savefig('network_parameters.png',
                bbox_inches='tight',
                pad_inches=0.5,  # 增加内边距
                dpi=150)  # 适当降低DPI以减小文件大小
    plt.close()  # 显式关闭图形释放内存


if __name__ == "__main__":
    visualize_network_parameters('best_model.pkl')
    print("可视化完成，结果已保存为 network_parameters.png")