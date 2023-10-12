# Single neuron model converted to output layer model
# Values from either input or other neurons
inputs = [1., 2., 3., 2.5]

# You can imagine they are the lines from inputs or other neurons
weights = [[0.2, 0.8, -0.5, 1.],
            [0.5, -0.91, 0.26, -.5],
            [-0.26, -0.27, 0.17, 0.87]]

# You can imagine this is the neuron itself, always only 1 for single neuron.
biases = [2, 3, 0.5]

layer_outputs = []  # Output for current layer

# Use zip to combine a matrix from weights and biases
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # Define output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

# Just to test zip
print("Tuple of weights and biases zipped: ", *zip(weights, biases))  # unpacks the zip iterator into a tuple
print("List of weights and biases zipped: ", [*zip(weights, biases)])  # unpacks to list, can also use list()

# List of lists of lists, 3D Array
lolol = [[[1,5,6,2],
          [3,2,1,3]],
         [[5,2,1,2],
          [6,4,8,4]],
         [[2,8,5,3],
          [1,1,9,4]]]
print("1D Array & Vector with shape(4): ", lolol[0][0])
print("2D Array & Matrix with shape(2, 4): ", lolol[0])
print("3D Array & Matrix with shape(3, 2, 4): ", lolol)
print("Tensor is an object that CAN be represented as a list.")
print("\n")
print("layer_outputs: ", layer_outputs)