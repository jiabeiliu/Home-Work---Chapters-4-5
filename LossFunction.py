import numpy as np
class LossFunction:
    def mse(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def mse_derivative(self, predicted, actual):
        return 2 * (predicted - actual) / actual.size
