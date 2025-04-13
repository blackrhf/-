import numpy as np
import matplotlib.pyplot as plt
from train import train_model, load_cifar10, preprocess_data
from test import evaluate_model


def hyperparameter_search(data_dir):
    """超参数网格搜索（带可视化）"""
    # 定义搜索范围
    hidden_sizes = [128, 256, 512]
    learning_rates = [0.1, 0.01, 0.001]
    reg_lambdas = [0.001, 0.01, 0.1]

    # 存储结果
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle("Hyperparameter Search Results", fontsize=14)

    # 网格搜索
    for hsize in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_lambdas:
                print(f"\nTesting hsize={hsize}, lr={lr:.4f}, reg={reg:.4f}")
                model = train_model(
                    data_dir=data_dir,
                    hidden_size=hsize,
                    learning_rate=lr,
                    reg_lambda=reg,
                    epochs=30  # 减少epochs以加快搜索
                )
                X_test, y_test = load_test_data(data_dir)
                test_acc = evaluate_model(model, X_test, y_test)
                results.append({
                    'hidden_size': hsize,
                    'learning_rate': lr,
                    'reg_lambda': reg,
                    'test_acc': test_acc
                })
                print(f"Test Accuracy: {test_acc:.4f}")

    # 可视化
    plot_results(results, axes)
    plt.tight_layout()
    plt.savefig('hyperparam_results.png')  # 保存图像
    plt.show()

    # 输出最佳参数
    best = max(results, key=lambda x: x['test_acc'])
    print("\n🔥 Best Parameters:")
    for k, v in best.items():
        print(f"{k:>15}: {v}")


def plot_results(results, axes):
    """绘制参数-准确率曲线"""
    # 转换为结构化数组便于筛选
    res_arr = np.array([(r['hidden_size'], r['learning_rate'], r['reg_lambda'], r['test_acc'])
                        for r in results],
                       dtype=[('hsize', int), ('lr', float), ('reg', float), ('acc', float)])

    # 1. 隐藏层大小 vs 准确率
    for hsize in np.unique(res_arr['hsize']):
        mask = res_arr['hsize'] == hsize
        axes[0].scatter(res_arr['lr'][mask], res_arr['acc'][mask],
                        label=f'hsize={hsize}', s=100 * res_arr['reg'][mask])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate (log scale)')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].legend()
    axes[0].set_title('Impact of Hidden Size & Learning Rate')

    # 2. 正则化强度 vs 准确率
    for reg in np.unique(res_arr['reg']):
        mask = res_arr['reg'] == reg
        axes[1].scatter(res_arr['hsize'][mask], res_arr['acc'][mask],
                        label=f'reg={reg:.3f}', s=50 + 1000 * res_arr['lr'][mask])
    axes[1].set_xlabel('Hidden Layer Size')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].legend()
    axes[1].set_title('Impact of Regularization')

    # 3. 三维参数关系
    sc = axes[2].scatter(
        res_arr['hsize'], np.log10(res_arr['lr']), c=res_arr['acc'],
        s=50 + 100 * res_arr['reg'], cmap='viridis', alpha=0.8
    )
    axes[2].set_xlabel('Hidden Size')
    axes[2].set_ylabel('log10(Learning Rate)')
    axes[2].set_title('3-Way Interaction')
    plt.colorbar(sc, ax=axes[2], label='Test Accuracy')


def load_test_data(data_dir):
    """加载测试数据"""
    _, _, X_test, y_test = load_cifar10(data_dir)
    return preprocess_data(X_test, y_test)


if __name__ == "__main__":
    hyperparameter_search('cifar-10-batches-py')