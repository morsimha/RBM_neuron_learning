import numpy as np
from sklearn.datasets import load_iris

# The sigmoid function takes an input 'x' and returns the sigmoid of 'x'.
# The sigmoid function is defined as 1 / (1 + exp(-x)), where exp is the exponential function.
# it maps any real-valued number into the range (0, 1), making it useful for probability estimation.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

orig_iris = load_iris()
iris = load_iris()
print("before:" ,iris.data[0])

# Discretize each attribute of every flower into one of three options: short, medium, long
def discretize_attributes(data):
    discretized_data = np.zeros_like(data)
    names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"]
    for i in range(data.shape[1]):
        print(names[i])
        column = data[:, i]
        thresholds = np.percentile(column, [33.33, 66.67])
        print("thresholds:", thresholds)
        discretized_data[:, i] = np.digitize(column, thresholds)
    return discretized_data

iris.data = discretize_attributes(iris.data)

# Print each attribute's original values and their discretized values
for i in range(iris.data.shape[1]):
    original_column = orig_iris.data[:, i]
    discretized_column = iris.data[:, i]
    print(f"Attribute {i + 1} original values: {original_column}")
    print(f"Attribute {i + 1} discretized values: {discretized_column}")

print(iris.data[5])
input_sample = iris.data[5]

# # Normalizing the input sample
# input_sample = (input_sample - input_sample.min()) / (input_sample.max() - input_sample.min())

#sets the seed for NumPy's random number generator to 42, ensuring that the results are reproducible.
np.random.seed(42)  # הגדרת seed כדי שהתוצאות יהיו שחזורות
# 3 neurons in the hidden layer
weights = np.random.rand(input_sample.shape[0], 3)  # נניח שיש לנו 3 נוירונים בשכבה החבויה

# Randomize the hidden layer values to be between 0 and 2
hidden_layer = np.random.randint(0, 3, size=(input_sample.shape[0], 3))

# # calculate the hidden layer by taking the dot product of the input sample and the weights, and then applying the sigmoid function.
# hidden_layer = sigmoid(np.dot(input_sample, weights))

# # calculate the reconstructed input by taking the dot product of the hidden layer and the weights, and then applying the sigmoid function.
# reconstructed_input = sigmoid(np.dot(hidden_layer, weights.T))

# # calculate the mean squared error (MSE) between the input sample and the reconstructed input.
# mse = np.mean((input_sample - reconstructed_input) ** 2)

print("Input Sample:", input_sample)
print("Weights:", weights)
print('\n')
print("Hidden Layer:", hidden_layer)
# print("Reconstructed Input:", reconstructed_input)
# print("Mean Squared Error (MSE):", mse)
