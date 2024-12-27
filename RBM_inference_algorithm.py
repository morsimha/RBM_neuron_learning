import numpy as np
from sklearn.datasets import load_iris
from RBM_visualizer import draw_rbm_network
import sys

# Energy Function
# E(v, h) = -sum(a_i * v_i) - sum(b_j * h_j) - sum(v_i * h_j * W_ij)
# the left and right synapses are calculated separately
def energy(v, h, o, a, b, c, Ji, Jk):
    term1 = np.sum(a * v)  #  visible neurons * Bias
    term1_1 = np.sum(c * o)  #  outputs visible neurons * Bias
    term2 = np.sum(b * h)  #  hidden neurons * Bias
    term3 = np.sum(v @ Ji * h) + np.sum(Jk @ o * h) # Interaction between synapses, visible and hidden neurons
    return -term1 - term1_1 - term2 - term3

orig_iris = load_iris()
iris = load_iris()

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

iris.data = discretize_attributes(iris.data)
input_sample = iris.data[5]

visible_neurons_amount = input_sample.shape[0]
hidden_neurons_amount = 12
output_neurons_amount = 3

# Initialize biases to small random values
visible_bias = np.random.normal(0, 0.1, visible_neurons_amount)
hidden_bias = np.random.normal(0, 0.1, hidden_neurons_amount)
output_bias = np.random.normal(0, 0.1, output_neurons_amount)

# Normalize synapse weights
left_synapses = np.random.normal(0, 0.1, (visible_neurons_amount, hidden_neurons_amount))
right_synapses = np.random.normal(0, 0.1, (hidden_neurons_amount, output_neurons_amount))


visible = input_sample
hidden = np.random.randint(0, 2, size=hidden_neurons_amount)
output = np.random.randint(0, 2, size=output_neurons_amount)

Temprature = 1
temp_decay = 0.99  # Slow decay to allow more randomness initially
iterations = 1000
energy_threshold = 0.01
neuron_change_threshold = 1
energy_change_threshold = 0.01
energy_change_window = 2

previous_energy = energy(visible, hidden, output, visible_bias, hidden_bias, output_bias, left_synapses, right_synapses)
energy_changes = []

print("Initial Visible State:", visible)
print("Initial Hidden State:", hidden)
print("Initial Output State:", output)
draw_rbm_network(visible, hidden, output, left_synapses, right_synapses)

for iteration in range(iterations):
    hidden_changes = 0
    output_changes = 0
    
    # Hidden Layer Update (Steps A-D)
    for j in range(hidden_neurons_amount):
        delta_E = hidden_bias[j] + np.sum(left_synapses[:, j] * visible)
        Pk = 1 / (1 + np.exp(-delta_E / Temprature))
        Xk = 1 if np.random.rand() < Pk else 0
        if hidden[j] != Xk:
            hidden_changes += 1
        hidden[j] = Xk
    
    # Output Layer Update (Steps A-D)
    for k in range(output_neurons_amount):
        delta_E = output_bias[k] + np.sum(right_synapses[:, k] * hidden)
        Pk = 1 / (1 + np.exp(-delta_E / Temprature))
        Xk = 1 if np.random.rand() < Pk else 0
        if output[k] != Xk:
            output_changes += 1
        output[k] = Xk
    
    current_energy = energy(visible, hidden, output, visible_bias, hidden_bias, output_bias, left_synapses, right_synapses)
    energy_changes.append(abs(previous_energy - current_energy))
    previous_energy = current_energy
   # draw_rbm_network(visible, hidden, output, left_synapses, right_synapses)

    
    print(f"Iteration {iteration + 1}, Energy: {current_energy}, Temperature: {Temprature}")
    Temprature *= temp_decay
    print("Current Output State:", output)
    
    if Temprature < 0.01:
        break
    
    if hidden_changes + output_changes < neuron_change_threshold:
        print("Stopping early due to small number of neuron changes.")
        break
    
    if len(energy_changes) > energy_change_window:
        recent_energy_changes = energy_changes[-energy_change_window:]
        if max(recent_energy_changes) < energy_change_threshold:
            print("Stopping early due to small energy changes.")
            break

print("\nFinal Visible State:", visible)
print("Final Hidden State:", hidden)
print("Final Output State:", output)
print(f"Final Energy: {current_energy}")

species_mapping = {
    (1, 0, 0): 'setosa',
    (0, 1, 0): 'versicolor',
    (0, 0, 1): 'virginica'
}

predicted_species = species_mapping.get(tuple(output), 'Unknown')
print(f"Predicted Iris Species: {predicted_species}")

draw_rbm_network(visible, hidden, output, left_synapses, right_synapses)
