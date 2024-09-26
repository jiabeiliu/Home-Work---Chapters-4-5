# Home-Work---Chapters-4-5
1. What is the Hadamard Matrix Product?
The Hadamard matrix product (also known as the element-wise product or Schur product) is a binary operation that takes two matrices of the same dimension and produces another matrix of the same dimension, where each element is the product of the corresponding elements of the input matrices. Mathematically, if 
𝐴
A and 
𝐵
B are two matrices of the same dimension, the Hadamard product 
𝐴
∘
𝐵
A∘B is:

(
𝐴
∘
𝐵
)
𝑖
𝑗
=
𝐴
𝑖
𝑗
×
𝐵
𝑖
𝑗
(A∘B) 
ij
​
 =A 
ij
​
 ×B 
ij
​
 
It is different from the traditional matrix multiplication (dot product), as it multiplies elements position-wise rather than across rows and columns.

2. Describe Matrix Multiplication
Matrix multiplication is the process of multiplying two matrices by taking the dot product of rows and columns. If matrix 
𝐴
A is of size 
𝑚
×
𝑛
m×n and matrix 
𝐵
B is of size 
𝑛
×
𝑝
n×p, their product 
𝐶
=
𝐴
×
𝐵
C=A×B will be a matrix of size 
𝑚
×
𝑝
m×p, where each element 
𝐶
𝑖
𝑗
C 
ij
​
  is calculated as:

𝐶
𝑖
𝑗
=
∑
𝑘
=
1
𝑛
𝐴
𝑖
𝑘
×
𝐵
𝑘
𝑗
C 
ij
​
 = 
k=1
∑
n
​
 A 
ik
​
 ×B 
kj
​
 
In matrix multiplication, the number of columns in the first matrix must equal the number of rows in the second matrix.

3. What is Transpose Matrix and Vector?
A transpose of a matrix is obtained by flipping the matrix over its diagonal. This means the rows become columns and the columns become rows. If 
𝐴
A is an 
𝑚
×
𝑛
m×n matrix, its transpose 
𝐴
𝑇
A 
T
  will be an 
𝑛
×
𝑚
n×m matrix. Mathematically:
(
𝐴
𝑇
)
𝑖
𝑗
=
𝐴
𝑗
𝑖
(A 
T
 ) 
ij
​
 =A 
ji
​
 
A transpose of a vector changes a row vector into a column vector or vice versa. For example, if 
𝑣
v is a column vector, its transpose 
𝑣
𝑇
v 
T
  will be a row vector.
4. Describe the Training Set Batch
In the context of training a neural network, a batch refers to a subset of the training data. Instead of passing the entire training set to the model at once, data is divided into batches, and each batch is passed through the network to update weights.

Mini-batch training is commonly used, where a small portion (batch) of the dataset is used in each iteration. For example, in stochastic gradient descent (SGD), weights are updated based on each mini-batch rather than the full dataset.
The size of the batch (batch size) controls how many examples the network processes before updating the model’s parameters.
Batch size affects the stability and speed of training. A large batch size smoothens the gradient but is computationally expensive, while a small batch size is faster but leads to noisier updates.
5. Describe the Entropy-based Loss (Cost or Error) Function and Explain Why it is Used for Training Neural Networks
The entropy-based loss function most commonly refers to the cross-entropy loss, which is used for classification tasks in neural networks. Cross-entropy measures the difference between two probability distributions, typically the predicted distribution from the neural network and the actual distribution (one-hot encoded labels).

For binary classification, the binary cross-entropy is used:

Loss
=
−
1
𝑁
∑
𝑖
=
1
𝑁
(
𝑦
𝑖
log
⁡
(
𝑝
𝑖
)
+
(
1
−
𝑦
𝑖
)
log
⁡
(
1
−
𝑝
𝑖
)
)
Loss=− 
N
1
​
  
i=1
∑
N
​
 (y 
i
​
 log(p 
i
​
 )+(1−y 
i
​
 )log(1−p 
i
​
 ))
For multi-class classification, the categorical cross-entropy is used:

Loss
=
−
∑
𝑖
𝑦
𝑖
log
⁡
(
𝑝
𝑖
)
Loss=− 
i
∑
​
 y 
i
​
 log(p 
i
​
 )
Where 
𝑦
𝑖
y 
i
​
  is the true label, and 
𝑝
𝑖
p 
i
​
  is the predicted probability of the corresponding class.

Why Cross-Entropy Loss is Used:

Cross-entropy loss effectively penalizes confident but wrong predictions.
It is sensitive to small differences in probabilities, which helps the model make better predictions.
It is a differentiable function, allowing gradient-based optimization algorithms (e.g., backpropagation) to be used for model training.
6. Describe Neural Network Supervised Training Process
In supervised training of a neural network, the model learns from labeled data. The process involves:

Forward Propagation: Input data is passed through the network, layer by layer, using the weights of the neurons to compute activations at each layer. The output layer generates predictions based on the activations.

Loss Computation: The difference between the network's predicted outputs and the actual target values (labels) is calculated using a loss function (e.g., cross-entropy).

Backpropagation: Gradients of the loss function with respect to each weight are calculated by applying the chain rule, starting from the output layer and propagating backward through the network.

Weight Update: Using an optimization algorithm like stochastic gradient descent (SGD), the weights are updated to reduce the loss. This involves taking a small step in the direction opposite to the gradient.

Repeat: This process is repeated for many iterations (epochs) until the model's predictions improve and the loss function converges.

7. Describe in Detail Forward Propagation and Backpropagation
Forward Propagation:
Forward propagation is the process by which the input data is passed through the network and transformed into output predictions. It involves:

Input Layer: The raw data is fed into the network.

Hidden Layers: Each layer computes the weighted sum of inputs from the previous layer, applies an activation function (e.g., ReLU, sigmoid), and passes the result to the next layer.

For each layer 
𝑙
l:

𝑧
(
𝑙
)
=
𝑊
(
𝑙
)
𝑎
(
𝑙
−
1
)
+
𝑏
(
𝑙
)
z 
(l)
 =W 
(l)
 a 
(l−1)
 +b 
(l)
 
where 
𝑊
(
𝑙
)
W 
(l)
  is the weight matrix, 
𝑎
(
𝑙
−
1
)
a 
(l−1)
  are the activations from the previous layer, and 
𝑏
(
𝑙
)
b 
(l)
  is the bias term.

Output Layer: The final layer produces the output predictions, usually through an activation function like softmax for classification.

Backpropagation:
Backpropagation is the process of calculating the gradient of the loss function with respect to each weight in the network. It involves:

Compute the Output Layer Gradient: The gradient of the loss with respect to the output layer’s weights is calculated. This is where the derivative of the loss function (e.g., cross-entropy) is applied.

𝛿
(
𝐿
)
=
∂
Loss
∂
𝑧
(
𝐿
)
=
𝑎
(
𝐿
)
−
𝑦
δ 
(L)
 = 
∂z 
(L)
 
∂Loss
​
 =a 
(L)
 −y
where 
𝑎
(
𝐿
)
a 
(L)
  is the activation of the output layer, and 
𝑦
y is the true label.

Backpropagate the Error: The error is propagated backward through the network to compute gradients for the weights in each previous layer. The chain rule is applied to compute how the loss changes with respect to each weight.

For each hidden layer 
𝑙
l, the error 
𝛿
(
𝑙
)
δ 
(l)
  is propagated backward:

𝛿
(
𝑙
)
=
(
𝑊
(
𝑙
+
1
)
)
𝑇
𝛿
(
𝑙
+
1
)
∘
𝜎
′
(
𝑧
(
𝑙
)
)
δ 
(l)
 =(W 
(l+1)
 ) 
T
 δ 
(l+1)
 ∘σ 
′
 (z 
(l)
 )
where 
∘
∘ denotes the Hadamard product (element-wise multiplication), and 
𝜎
′
(
𝑧
(
𝑙
)
)
σ 
′
 (z 
(l)
 ) is the derivative of the activation function.

Update the Weights: Once the gradients are calculated, the weights are updated using an optimization method like gradient descent:

𝑊
(
𝑙
)
=
𝑊
(
𝑙
)
−
𝜂
∂
Loss
∂
𝑊
(
𝑙
)
W 
(l)
 =W 
(l)
 −η 
∂W 
(l)
 
∂Loss
​
 
where 
𝜂
η is the learning rate.
