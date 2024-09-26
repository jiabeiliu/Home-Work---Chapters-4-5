import numpy as np
from Model import Model
from Training import Training
from LossFunction import LossFunction
from GradDescent import GradDescent
import sys
# 初始化模型、损失函数、优化器
model = Model(n_inputs=2, n_hidden_neurons=5, n_output_neurons=1)
loss_function = LossFunction()
optimizer = GradDescent()

# 创建训练器并开始训练
trainer = Training(model, loss_function, optimizer)

X = np.array([[0.1, 0.2],   # 1st sample, 2 features
              [0.4, 0.5],   # 2nd sample, 2 features
              [0.7, 0.8]])  # 3rd sample, 2 features
# Shape: (3, 2), 3 samples, 2 features per sample

Y = np.array([[1],    # Target output for 1st sample
              [0],    # Target output for 2nd sample
              [1]])   # Target output for 3rd sample
# Shape: (3, 1), target outputs for 3 samples

trainer.train(X, Y, epochs=100, learning_rate=0.1)
print(sys.path)