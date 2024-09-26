import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))  # 偏置初始化为0
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-self.z))
        return self.output

    def backward(self, grad_output, learning_rate):
        # 计算激活函数的梯度
        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.z))
            grad_z = grad_output * sigmoid * (1 - sigmoid)

        # 计算损失对权重和偏置的梯度
        grad_weights = np.dot(self.inputs.T, grad_z)
        grad_biases = np.sum(grad_z, axis=0, keepdims=True)

        # 更新权重和偏置
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        # 返回传递给前一层的梯度
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
