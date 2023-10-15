import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# Applied to each value in array
exp_values = np.exp(layer_outputs)

print(exp_values)

# Normalization

print()
# default, sum off all features
print(np.sum(layer_outputs, axis=None))

# sums of columns in matrix
print(np.sum(layer_outputs, axis=0))

# sums of rows in matrix
print(np.sum(layer_outputs, axis=1))

# keep the dimensions to line it up
print(np.sum(layer_outputs, axis=1, keepdims=True))
print()


norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)