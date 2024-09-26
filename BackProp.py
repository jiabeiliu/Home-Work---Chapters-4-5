class BackProp:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function

    def backward(self, inputs, actual_output, predicted_output, learning_rate=0.01):
        # Implement backpropagation here
        pass
