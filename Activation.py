import numpy as np
class Activation:
    def __init__(self, activation_type='relu'):
        self.activation_type = activation_type
        self.activation_functions = {
            'relu': (self.relu, self.relu_derivative),
            'sigmoid': (self.sigmoid, self.sigmoid_derivative)
        }

    def forward(self, z):
        return self.activation_functions[self.activation_type][0](z)

    def derivative(self, z):
        return self.activation_functions[self.activation_type][1](z)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)
