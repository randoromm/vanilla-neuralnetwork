# Single neuron model
# Values from either input or other neurons
inputs = [1., 2., 3., 2.5]

# You can imagine they are the lines from inputs or other neurons
weights = [0.2, 0.8, -0.5, 1.]

# You can imagine this is the neuron itself, always only 1 for single neuron.
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output)
