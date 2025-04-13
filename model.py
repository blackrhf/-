import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size=3072, hidden_size=256, output_size=10, activation='relu'):
        """
        三层神经网络初始化
        参数:
            input_size: 输入维度 (CIFAR-10展平后为3072)
            hidden_size: 隐藏层神经元数量
            output_size: 输出类别数 (CIFAR-10为10)
            activation: 激活函数 ('relu'或'sigmoid')
        """
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size),  # He初始化
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size),
            'b2': np.zeros(output_size)
        }
        self.activation = activation
    
    def forward(self, X):
        """前向传播"""
        self.cache = {}
        # 第一层 (全连接 + 激活)
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        if self.activation == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        self.cache['Z1'], self.cache['A1'] = Z1, A1
        
        # 第二层 (全连接 + Softmax)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        exp_scores = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.cache['Z2'], self.cache['A2'] = Z2, A2
        return A2
    
    def backward(self, X, y, reg_lambda=0.0):
        """反向传播计算梯度"""
        m = X.shape[0]
        grads = {}
        A1, A2 = self.cache['A1'], self.cache['A2']
        W1, W2 = self.params['W1'], self.params['W2']
        
        # 输出层梯度
        dZ2 = A2 - y
        grads['W2'] = A1.T.dot(dZ2) / m + reg_lambda * W2
        grads['b2'] = np.sum(dZ2, axis=0) / m
        
        # 隐藏层梯度
        dA1 = dZ2.dot(W2.T)
        if self.activation == 'relu':
            dZ1 = dA1 * (A1 > 0)
        elif self.activation == 'sigmoid':
            dZ1 = dA1 * A1 * (1 - A1)
        grads['W1'] = X.T.dot(dZ1) / m + reg_lambda * W1
        grads['b1'] = np.sum(dZ1, axis=0) / m
        return grads
    
    def compute_loss(self, X, y, reg_lambda=0.0):
        """计算交叉熵损失 + L2正则化"""
        m = y.shape[0]
        A2 = self.forward(X)
        corect_logprobs = -np.log(A2[range(m), y.argmax(axis=1)])
        data_loss = np.sum(corect_logprobs) / m
        reg_loss = 0.5 * reg_lambda * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss