import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

'''
Now we have batches of data:
imagine class targets are:
dog = 0
cat = 1
human = 2

The following class targets would mean the first sample in the batch would be for index 0,
the second sample in the batch would be for index 1,
the third sample in the batch would be again for index 1.

so for [0.7, 0.1, 0.2] we use index 0 (as said in class_targets) for the prediction and calculate loss based on that.
'''
class_targets = [0, 1, 1]

# Zipping using numpy (Result is [0.7, 0.5, 0.9]):
# softmax_outputs[[*first dimension indexes we are interested in*], [*second dimension indexes*]])
print(softmax_outputs[[0, 1, 2], class_targets])

# Remove hard coding
print(softmax_outputs[range(len(softmax_outputs)), class_targets])

loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(f"loss: {loss}")

average_loss = np.mean(loss)
print(f"mean loss: {average_loss}")

# Calculating accuracy (will revisit later):
# Get the indexes where the prediction was the highest in a sample
predictions = np.argmax(softmax_outputs, axis=1)
# Mean of matches between predictions and class targets
accuracy = np.mean(predictions == class_targets)
print("acc:", accuracy)

"""
The issue:
log of 0 is infinite and we get an exception. If one of the predictions is 0 we are in trouble.

"""
print()
print("The issue with 0 as a prediction (although not entirely wrong):")
softmax_outputs = np.array([[0.0, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]
loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(loss)
print(f"loss: {loss}")
print(f"Average_loss: {average_loss}")
print("Therefore we need clipping of predictions (check the full example)")

print("Accuracy, when 1 out of 3 was wrong:")
predictions = np.argmax(softmax_outputs, axis=1)
accuracy = np.mean(predictions == class_targets)
print("acc:", accuracy)