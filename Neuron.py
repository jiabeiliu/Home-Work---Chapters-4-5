import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs, 1) * 0.01  # Small random values
 # Weights shape (n_inputs, 1)
        self.bias = np.random.randn()

    def forward(self, inputs):
        inputs = np.atleast_2d(inputs)  # Ensure inputs are 2D, shape (n_samples, n_inputs)
        return np.dot(inputs, self.weights) + self.bias  # Outputs shape (n_samples, 1)
