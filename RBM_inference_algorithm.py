import numpy as np
from sklearn.datasets import load_iris

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Energy Function
def energy(v, h, o, a, b, c, J, W):
    """
    Calculate the energy of the Boltzmann Machine.
    v: Visible layer (array of visible neuron states)
    h: Hidden layer (array of hidden neuron states)
    o: Output layer (array of output neuron states)
    a: Biases for visible neurons
    b: Biases for hidden neurons
    c: Biases for output neurons
    J: Weights between visible and hidden neurons
    W: Weights between hidden and output neurons
    """
    term1 = -np.sum(a * v)  # Bias of visible neurons
    term2 = -np.sum(b * h)  # Bias of hidden neurons
    term3 = -np.sum(c * o)  # Bias of output neurons
    term4 = -np.sum(v @ J * h)  # Interaction between visible and hidden neurons
    term5 = -np.sum(h @ W * o)  # Interaction between hidden and output neurons
    
    return term1 + term2 + term3 + term4 + term5

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
visible_neurons_amount = input_sample.shape[0]  # 4 visible neurons (4 attributes)
hidden_neurons_amount = 3  # 3 hidden neurons
output_neurons_amount = 3  # 3 output neurons for species

# Initialize biases and weights
visible_bias = np.random.rand(visible_neurons_amount)  # Bias for visible neurons
hidden_bias = np.random.rand(hidden_neurons_amount)   # Bias for hidden neurons
output_bias = np.random.rand(output_neurons_amount)   # Bias for output neurons

left_synapses = np.random.rand(visible_neurons_amount, hidden_neurons_amount)  # Weights between visible and hidden
right_synapses = np.random.rand(hidden_neurons_amount, output_neurons_amount)  # Weights between hidden and output

# Initialize neuron states randomly
visible = input_sample  # Visible neurons set to the input sample
hidden = np.random.randint(0, 2, size=hidden_neurons_amount)  # Hidden neurons initialized randomly (binary states)
output = np.random.randint(0, 2, size=output_neurons_amount)  # Output neurons initialized randomly (binary states)

# Print biases and synapse values
print("Visible Neuron Biases (a):")
print(visible_bias)
print("\nHidden Neuron Biases (b):")
print(hidden_bias)
print("\nOutput Neuron Biases (c):")
print(output_bias)
print("\nLeft Synapse Weights (J):")
print(left_synapses)
print("\Right Synapse Weights (W):")
print(right_synapses)

# Inference Algorithm
Temprature = 1.0  # Initial temperature
temp_decay = 0.95  # Temperature decay factor
iterations = 1000

for iteration in range(iterations):
    # Update hidden neurons
    for j in range(hidden_neurons_amount):
        delta_E = hidden_bias[j] + np.sum(left_synapses[:, j] * visible)
        hidden[j] = 1 if np.random.rand() < sigmoid(delta_E / Temprature) else 0
    
    # Update output neurons
    for k in range(output_neurons_amount):
        delta_E = output_bias[k] + np.sum(right_synapses[:, k] * hidden)
        output[k] = 1 if np.random.rand() < sigmoid(delta_E / Temprature) else 0
    
    # # Update visible neurons
    # for i in range(visible_neurons_amount):
    #     delta_E = a[i] + np.sum(J[i, :] * h)
    #     v[i] = 1 if np.random.rand() < sigmoid(delta_E / T) else 0
    
    # Calculate current energy
    current_energy = energy(visible, hidden, output, visible_bias, hidden_bias, output_bias, left_synapses, right_synapses)
    print(f"Iteration {iteration + 1}, Energy: {current_energy}, Temperature: {Temprature}")
    
    # Reduce temperature
    Temprature *= temp_decay

    # print("current Visible State:", visible)
    # print("current Hidden State:", hidden)
    print("current Output State:", output)
    
    # Check convergence (simple energy stabilization check)
    if Temprature < 0.01:
        break

print("\nFinal Visible State:", visible)
print("Final Hidden State:", hidden)
print("Final Output State:", output)
print(f"Final Energy: {current_energy}")

# Prediction Step
species_mapping = {
    (1, 0, 0): 'setosa',
    (0, 1, 0): 'versicolor',
    (0, 0, 1): 'virginica'
}

predicted_species = species_mapping.get(tuple(output), 'Unknown')
print(f"Predicted Iris Species: {predicted_species}")
