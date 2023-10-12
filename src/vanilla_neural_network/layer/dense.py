import numpy as np


class Dense:
    """
    This module is for initializing dense layer object and working with the said layer.
    """
    def __init__(self, n_inputs, n_neurons):
        """
        Initializes the dense layer.

        Generally we want to keep the weights around -0.1 to 0.1 in the beginning. During the development of the network
        otherwise things can explode with all the multiplication. Good idea is to also normalize the inputs between -1 to 1.
        Biases are usually kept at 0 by default. If you notice that the outputs are not generated, it might mean that
        the network is dead. Then adding non-zero bias might make sense.

        :param n_inputs: (int) Number of inputs for this layer
        :param n_neurons: (int) Number of neurons in this layer
        """
        # Initialising weights the other way than before.
        # This way we avoid 1 extra transpose every time we do a forward.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # Expects shape in separate parameters
        self.biases = np.zeros((1, n_neurons))  # Expects a shape on 1st argument, there4 tuple
        self.output = None

    def forward(self, inputs):
        """
        Calculates dot product on given inputs and weights of the layer and adds biases.
        :param inputs: Inputs for this layer.
        :return: None
        """
        self.output = np.dot(inputs, self.weights) + self.biases
