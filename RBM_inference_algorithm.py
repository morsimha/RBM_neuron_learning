import numpy as np
from sklearn.datasets import load_iris
from RBM_visualizer import draw_rbm_network
from utils import discretize_attributes, energy, initialize_random_parameters

Temprature = 1
temp_decay = 0.99  # Slow decay to allow more randomness initially
iterations = 1000
energy_threshold = 0.01
neuron_change_threshold = 1
energy_change_threshold = 0.01
energy_change_window = 2

visible_neurons_amount = 8 # Based on discretized Iris data
hidden_neurons_amount = 12 # the idea was that each visible neuron would project to a fitting hidden neuron 
output_neurons_amount = 3 # One for each Iris species

orig_iris = load_iris()
iris = load_iris()

iris.data = discretize_attributes(iris.data)

visible_bias, hidden_bias, output_bias, left_synapses, right_synapses = initialize_random_parameters(
    visible_neurons_amount, hidden_neurons_amount, output_neurons_amount)

input_sample = iris.data[5]
print("Input Sample - iris.data[5]:", orig_iris.data[5])
visible = input_sample

hidden = np.random.randint(0, 2, size=hidden_neurons_amount)
output = np.random.randint(0, 2, size=output_neurons_amount)

previous_energy = energy(visible, hidden, output, visible_bias, hidden_bias, output_bias, left_synapses, right_synapses)
energy_changes = []

print("Initial Visible State:", visible , output)
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
