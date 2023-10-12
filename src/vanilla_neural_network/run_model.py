import nnfs
from nnfs.datasets import spiral_data

from layer.dense import Dense
from activation.relu import ReLu


def run():
    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    dense1 = Dense(2, 3)
    activation1 = ReLu()
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