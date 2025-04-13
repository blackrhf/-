import pickle
import numpy as np
import matplotlib.pyplot as plt
from train import load_cifar10, preprocess_data


def evaluate_model(model, X, y):
    """评估模型性能"""
    # 计算预测结果
    probs = model.forward(X)
    y_pred = probs.argmax(axis=1)
    y_true = y.argmax(axis=1)

    # 计算准确率
    accuracy = np.mean(y_pred == y_true)

    # 计算损失（需使用模型自带的compute_loss方法）
    loss = model.compute_loss(X, y, reg_lambda=0.01)  # 使用训练时的正则化系数

    return accuracy, loss


def plot_class_accuracy(model, X, y, class_names):
    """绘制每个类别的准确率"""
    probs = model.forward(X)
    y_pred = probs.argmax(axis=1)
    y_true = y.argmax(axis=1)

    class_acc = []
    for i in range(10):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            class_acc.append(acc)

    plt.figure(figsize=(10, 5))
    plt.bar(range(10), class_acc)
    plt.xticks(range(10), class_names, rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Test Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.show()


if __name__ == "__main__":
    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载测试数据
    _, _, X_test, y_test = load_cifar10('cifar-10-batches-py')
    X_test, y_test = preprocess_data(X_test, y_test)

    # 加载模型
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 评估整体性能
    test_acc, test_loss = evaluate_model(model, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 绘制各类别准确率
    plot_class_accuracy(model, X_test, y_test, class_names)

