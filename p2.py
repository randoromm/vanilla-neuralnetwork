# Single neuron model converted to output layer model (or any other layer with 3 neurons)
# inputs - can be from input layer or other neurons outputs.
inputs = [1., 2., 3., 2.5]

# You can imagine they are the lines from inputs or other neurons
weights1 = [0.2, 0.8, -0.5, 1.]
weights2 = [0.5, -0.91, 0.26, -.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# You can imagine these are the neurons itself, always only 1 for single neuron.
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)