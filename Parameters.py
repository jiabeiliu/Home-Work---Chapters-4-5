class Parameters:
    def __init__(self, layers):
        self.layers = layers

    def get_weights_and_biases(self):
        return [(neuron.weights, neuron.bias) for layer in self.layers for neuron in layer.neurons]
