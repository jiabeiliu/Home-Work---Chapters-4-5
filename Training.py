class Training:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            # 前向传播
            predicted_output = self.model.forward(X)

            # 计算损失
            loss = self.loss_function.mse(predicted_output, Y)
            print(f"Epoch {epoch}, Loss: {loss}")

            # 反向传播
            # 计算损失相对于输出的梯度
            loss_grad = self.loss_function.mse_derivative(predicted_output, Y)

            # 反向传播输出层
            grad_out2 = self.model.layer2.backward(loss_grad, learning_rate)

            # 反向传播隐藏层
            grad_out1 = self.model.layer1.backward(grad_out2, learning_rate)

            # 更新权重
            self.optimizer.update(self.model, learning_rate)
