import numpy as np
from sklearn.datasets import load_iris

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Energy Function
def energy(v, h, a, b, J):
    """
    Calculate the energy of the Boltzmann Machine.
    v: Visible layer (array of visible neuron states)
    h: Hidden layer (array of hidden neuron states)
    a: Biases for visible neurons
    b: Biases for hidden neurons
    J: Weights between visible and hidden neurons
    """
    term1 = -np.sum(a * v)  # Bias of visible neurons
    term2 = -np.sum(b * h)  # Bias of hidden neurons
    term3 = -np.sum(v @ J * h)  # Interaction between visible and hidden neurons
    
    return term1 + term2 + term3

# Load Iris dataset
orig_iris = load_iris()
iris = load_iris()

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

# Select a sample from the dataset
input_sample = iris.data[5]

# Initialize parameters
np.random.seed(42)
visible_neurons = input_sample.shape[0]  # 4 visible neurons (4 attributes)
hidden_neurons = 3  # 3 hidden neurons as per the architecture

# Initialize biases and weights
a = np.random.rand(visible_neurons)  # Bias for visible neurons
b = np.random.rand(hidden_neurons)   # Bias for hidden neurons
J = np.random.rand(visible_neurons, hidden_neurons)  # Weights between visible and hidden

# Initialize neuron states randomly
v = input_sample  # Visible neurons set to the input sample
h = np.random.randint(0, 2, size=hidden_neurons)  # Hidden neurons initialized randomly (binary states)

# Print biases and synapse values
print("Visible Neuron Biases (a):")
print(a)
print("\nHidden Neuron Biases (b):")
print(b)
print("\nSynapse Weights (J):")
print(J)

# Calculate energy
current_energy = energy(v, h, a, b, J)
print(f"\nInitial Energy of the System: {current_energy}")
