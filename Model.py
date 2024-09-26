from Layer import Layer
class Model:
    def __init__(self, n_inputs, n_hidden_neurons, n_output_neurons):
        self.layer1 = Layer(n_inputs, n_hidden_neurons, 'relu')
        self.layer2 = Layer(n_hidden_neurons, n_output_neurons, 'sigmoid')

    def forward(self, inputs):
        out1 = self.layer1.forward(inputs)
        out2 = self.layer2.forward(out1)
        return out2
