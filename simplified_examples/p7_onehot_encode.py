"""
Calculating loss with onehot encoding:
Categorical Cross-Entropy:
We create a vector at the length of our number of classes.
Then we can calculate loss on our results of softmax predictions vector.


"""
import math

softmax_output = [0.7, 0.1, 0.2]
# target_class = 0
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
print(f"Full equation: {loss}")

loss = -(math.log(softmax_output[0]))

print(f"Simplified equation: {loss}")
print()
print(f"0.7 loss: {-(math.log(0.7))}")
print("As the prediction confidence goes lower the loss goes higher:")
print(f"0.5 loss: {-(math.log(0.5))}")
