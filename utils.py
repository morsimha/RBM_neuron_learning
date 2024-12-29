import numpy as np

# Energy Function
# Expanding the energy function from the book, to represent the separation between visible neurons
# E(v, h, o) = -sum(a_i * v_i) - sum(b_j * h_j) - sum(c_k * o_k) - sum(v_i * h_j * W_ij) - sum(h_j * o_k * W_jk)
def energy(v, h, o, a, b, c, Ji, Jk):
    term1 = np.sum(a * v)  # Visible neurons * Bias
    term2 = np.sum(b * h)  # Hidden neurons * Bias
    term3 = np.sum(c * o)  # Output neurons * Bias
    term4 = np.sum(v @ Ji * h)  # Interaction between visible and hidden neurons
    term5 = np.sum(h @ Jk * o)  # Interaction between hidden and output neurons
    return -term1 - term2 - term3 - term4 - term5

def discretize_attributes(data):
    discretized_data = np.zeros((data.shape[0], data.shape[1] * 2))
    names = ["sepal length", "sepal width", "petal length", "petal width"]
    for i in range(data.shape[1]):
        column = data[:, i]
        thresholds = np.percentile(column, [33.33, 66.67])
        categories = np.digitize(column, thresholds)
        for j, value in enumerate(categories):
            if value == 0:
                discretized_data[j, i * 2:i * 2 + 2] = [0, 0]
            elif value == 1:
                discretized_data[j, i * 2:i * 2 + 2] = [0, 1]
            else:
                discretized_data[j, i * 2:i * 2 + 2] = [1, 0]
    return discretized_data

def initialize_random_parameters(visible_neurons_amount, hidden_neurons_amount, output_neurons_amount):
    # Initialize biases to small random values
    visible_bias = np.random.normal(0, 0.1, visible_neurons_amount)
    hidden_bias = np.random.normal(0, 0.1, hidden_neurons_amount)
    output_bias = np.random.normal(0, 0.1, output_neurons_amount)
    
    # Normalize synapse weights
    # Initialize synapses to small random values inbetween -0.1 and 0.1
    left_synapses = np.random.normal(0, 0.1, (visible_neurons_amount, hidden_neurons_amount))
    right_synapses = np.random.normal(0, 0.1, (hidden_neurons_amount, output_neurons_amount))
    
    return visible_bias, hidden_bias, output_bias, left_synapses, right_synapses
