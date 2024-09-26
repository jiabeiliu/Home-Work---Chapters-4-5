class ForwardProp:
    def __init__(self, model):
        self.model = model

    def forward(self, inputs):
        return self.model.forward(inputs)
