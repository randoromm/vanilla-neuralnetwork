import numpy as np

'''
Introducing batch. Batch is useful, because we can multiprocess and mainly to generalize.
Imagine a vector/line that shows the average fitness above random inputs that are displayed every frame in graph
How easy would it be to draw the average line on the graph if 1 dot was flashing everywhere on the graph?
It would be jittery. If we upped the amount of input dots simultaneously to 4, it would be much easier to draw the line.
That's generalization.
'''
inputs = [[1., 2., 3., 2.5],
          [2.0, 5., -1., 2.],
          [-1.5, 2.7, 3.3, -.8]]

weights = [[0.2, 0.8, -0.5, 1.],
           [0.5, -0.91, 0.26, -.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2., 3., 0.5]

'''
At the first argument in np.dot(), the shape of Dimension 1 needs to match the shape of Dimension 0 of
the second argument in np.dot(). 
'''
print(f"Weights matrix after transform: \n{np.array(weights).T}, "
      f"\nShape of inputs: {np.array(inputs).shape}"
      f"\nShape of the weigthts after transpose: {np.array(weights).T.shape}")

# Output calculation
output = np.dot(np.array(inputs), np.array(weights).T) + biases

print(f"Output after dot product:\n{output}")