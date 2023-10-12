import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

'''
# Input feature sets capital x is a standard in ML - actual input data.
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
'''


class Layer_Dense:
    """
    Generally we want to keep the weights around -0.1 to 0.1 in the beginning. During the development of the network
    otherwise things can explode with all the multiplication. Good idea is to also normalize the inputs between -1 to 1.

    Biases are usually kept at 0 by default. If you notice that the outputs are not generated, it might mean that
    the network is dead. Then adding non-zero bias might make sense.
    """

    def __init__(self, n_inputs, n_neurons):
        # Initialising weights the other way than before.
        # This way we avoid 1 extra transpose every time we do a forward.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # Expects shape in separate parameters
        self.biases = np.zeros((1, n_neurons))  # Expects a shape on 1st argument, there4 tuple
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


'''
# The number of inputs is how many features are inside each sample
# Number of neurons can be whatever you choose.
layer1 = Layer_Dense(4, 5)
# On layer 2, the inputs must match the outputs of layer 1
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(f"Layer1 output: \n{layer1.output}")
layer2.forward(layer1.output)
print(f"Layer2 output: \n{layer2.output}")
'''

'''
Create a dataset with a function from nnfs package.
Dataset includes 300 points and their coordinates on a 2D graph (x, y).
As a duple ('(X, y)') each of the 300 points is assigned one of the 3 classes.
Shape Explanation:
    X has a shape of (300, 2), 300 points with x & y coordinates.
    y has a shape of (300), List of 300 class numbers (1, 2, 3)
'X' and 'y' are separate variables from 'spiral_data()' function.    
'''
X, y = spiral_data(samples=100, classes=3)
print(f"X shape: {X.shape}, y shape: {y.shape}")
dense1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = None
loss_function = None

print(f"Dense1 weights shape: {dense1.weights.shape}, dense1 biases shape: {dense1.weights.shape}")
dense1.forward(X)
print(f"Dense1 output shape: {dense1.output.shape}")
activation1.forward(dense1.output)
print(f"Activation1 output shape: {dense1.output.shape}")

print(activation1.output)
