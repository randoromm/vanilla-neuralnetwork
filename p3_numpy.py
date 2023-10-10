import numpy as np

inputs = [1., 2., 3., 2.5]

weights = [[0.2, 0.8, -0.5, 1.],
           [0.5, -0.91, 0.26, -.5],
           [-0.26, -0.27, 0.17, 0.87]]
weight_vec = weights[0]

biases = [2, 3, 0.5]
bias = biases[0]

# dot_product + bias of single neuron weights and inputs and bias
output_single_neuron = np.dot(weight_vec, inputs) + bias
print(output_single_neuron)

# Weights matrix, same thing:
'''
output = [np.dot(weights[0], inputs) + biases[0], 
          np.dot(weights[1], inputs) + biases[1], 
          np.dot(weights[2], inputs) + biases[2]]
'''
# Order is important. Matrix before Vector for dot. Otherwise shape error.
output = np.dot(weights, inputs) + biases
print(output)