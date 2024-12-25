import numpy as np
from sklearn.datasets import load_iris

# The sigmoid function takes an input 'x' and returns the sigmoid of 'x'.
# The sigmoid function is defined as 1 / (1 + exp(-x)), where exp is the exponential function.
# it maps any real-valued number into the range (0, 1), making it useful for probability estimation.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

iris = load_iris()
print(iris.data[0])
input_sample = iris.data[0]

# Normalizing the input sample
input_sample = (input_sample - input_sample.min()) / (input_sample.max() - input_sample.min())

#sets the seed for NumPy's random number generator to 42, ensuring that the results are reproducible.
np.random.seed(42)  # הגדרת seed כדי שהתוצאות יהיו שחזורות
# 3 neurons in the hidden layer
weights = np.random.rand(input_sample.shape[0], 3)  # נניח שיש לנו 3 נוירונים בשכבה החבויה


# calculate the hidden layer by taking the dot product of the input sample and the weights, and then applying the sigmoid function.
hidden_layer = sigmoid(np.dot(input_sample, weights))

# calculate the reconstructed input by taking the dot product of the hidden layer and the weights, and then applying the sigmoid function.
reconstructed_input = sigmoid(np.dot(hidden_layer, weights.T))

# calculate the mean squared error (MSE) between the input sample and the reconstructed input.
mse = np.mean((input_sample - reconstructed_input) ** 2)

print("Input Sample:", input_sample)
print("Weights:", weights)
print('\n')
print("Hidden Layer:", hidden_layer)
print("Reconstructed Input:", reconstructed_input)
print("Mean Squared Error (MSE):", mse)
