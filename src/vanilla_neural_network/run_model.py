import nnfs
from nnfs.datasets import spiral_data

from layer.dense import Dense
from activation.relu import ReLU


def run():
    # Sets a random seed and changed np.dot functions data type
    nnfs.init()

    '''
    For the spiral dataset,
    - features are the x-coordinates (x) and y-coordinates (y) of the points
            - In the code, there are 300 x and 300 y values associated with the 300 points
    - feature sets are the pairs (x, y) that fully define one point in the dataset
            - In the code, there are 300 feature sets 
    - classes are the labels associated to the points
            - In the code, there are 3 classes defined by the colors - red, blue, green - and each feature set (x, y) 
            corresponds to one of these 3 classes (with 100 points each)
    - samples are the combination of feature sets and classes that form the dataset
            - For example: (x = 0.2, y = -0.5, color = red) and (x = -0.5, y = -0.2, color=blue) are 
            samples from the dataset
    
    Calling the function X, y = spiral_data(100, 3) creates samples belonging to 3 classes with 100 feature 
    sets each. 
    X (feature sets) is an array of shape (300, 2) and y (classes) is a vector of size 300.
    
    Based on: https://cs231n.github.io/neural-networks-case-study/
    '''
    X, y = spiral_data(samples=100, classes=3)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    dense1 = Dense(2, 3)
    activation1 = ReLU()
    dense2 = Dense(3, 3)
    activation2 = None
    loss_function = None

    print(f"Dense1 weights shape: {dense1.weights.shape}, dense1 biases shape: {dense1.weights.shape}")
    dense1.forward(X)
    print(f"Dense1 output shape: {dense1.output.shape}")
    activation1.forward(dense1.output)
    print(f"Activation1 output shape: {dense1.output.shape}")


if __name__ == "__main__":
    run()