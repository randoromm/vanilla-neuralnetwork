import numpy as np


class ReLu:
    """
    This module is for applying ReLu Activation function on an input and storing an output in the object.
    """
    def __init__(self):
        """
        Currently only initiates the object and defines the empty output variable.
        """
        self.output = None

    def forward(self, inputs):
        """
        Performs ReLu activation on inputs and writes output to 'ReLu.output'.
        :param inputs: Input values as a ndarray.
        :return: None
        """
        self.output = np.maximum(0, inputs)