import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from model import ThreeLayerNN


def load_cifar10(data_dir):
    """加载CIFAR-10数据集"""

    def load_batch(file):
        with open(file, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        X = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(batch['labels'])
        return X, y

    train_batches = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 5)]
    test_batch = os.path.join(data_dir, 'test_batch')

    X_train, y_train = [], []
    for batch in train_batches:
        X, y = load_batch(batch)
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_batch(test_batch)
    return X_train, y_train, X_test, y_test


def preprocess_data(X, y, num_classes=10):
    """数据预处理"""
    X = X.reshape(X.shape[0], -1).astype('float32') / 255
    y = np.eye(num_classes)[y]
    return X, y


def train_model(data_dir, hidden_size=256, activation='relu',
                learning_rate=0.01, reg_lambda=0.01, epochs=100,
                batch_size=64, lr_decay=0.95):
    """训练模型并绘制训练曲线"""
    # 加载数据
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # 划分验证集
    val_size = 1000
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    # 初始化模型和记录器
    model = ThreeLayerNN(
        input_size=3072,
        hidden_size=hidden_size,
        output_size=10,
        activation=activation
    )
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # 训练循环
    best_val_acc = 0
    for epoch in range(epochs):
        if epoch % 10 == 0 and epoch > 0:
            learning_rate *= lr_decay

        # Mini-batch SGD
        indices = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            model.forward(X_batch)
            grads = model.backward(X_batch, y_batch, reg_lambda)

            for param in model.params:
                model.params[param] -= learning_rate * grads[param]

        # 计算训练集指标
        train_loss = model.compute_loss(X_train, y_train, reg_lambda)
        train_pred = model.forward(X_train).argmax(axis=1)
        train_acc = np.mean(train_pred == y_train.argmax(axis=1))

        # 验证集评估
        val_acc = np.mean(model.forward(X_val).argmax(axis=1) == y_val.argmax(axis=1))

        # 记录指标
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # 保存最优模型
    model.params = best_params
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    return model


if __name__ == "__main__":
    train_model(data_dir='cifar-10-batches-py')