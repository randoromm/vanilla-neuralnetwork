import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

# E = 2.71828182846
E = math.e

# Applied to each value in array
exp_values = np.exp(layer_outputs)

print(exp_values)

# Normalization
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))