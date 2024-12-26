import numpy as np
from sklearn.datasets import load_iris
import sys
# -----------------------------
# תנאים מוקדמים (Preconditions)
# -----------------------------
# 1. הפעלת אלגוריתם למידת מכונת בולצמן באופן מיטבי.
# 2. שימוש בפרמטרים a, b, J המתקבלים מאלגוריתם הלמידה.

# # Activation Function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# Energy Function
def energy(v, h, o, a, b, c, J, W):
    term1 = -np.sum(a * v)
    term2 = -np.sum(b * h)
    term3 = -np.sum(c * o)
    term4 = -np.sum(v @ J * h)
    term5 = -np.sum(h @ W * o)
    return term1 + term2 + term3 + term4 + term5

# -----------------------------
# אתחול (Initialization)
# -----------------------------
# 1. קלט הנתונים והכנת הפרמטרים הדרושים.
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

#np.random.seed(42)
visible_neurons_amount = input_sample.shape[0]
hidden_neurons_amount = 12
output_neurons_amount = 3

visible_bias = np.random.rand(visible_neurons_amount)
hidden_bias = np.random.rand(hidden_neurons_amount)
output_bias = np.random.rand(output_neurons_amount)

left_synapses = np.random.rand(visible_neurons_amount, hidden_neurons_amount)
print(left_synapses)

# left_synapses = np.zeros((visible_neurons_amount, hidden_neurons_amount))
# for i in range(visible_neurons_amount):
#     if i < 3:
#         left_synapses[i, :3] = np.random.rand(3)
#     elif 3 <= i < 6:
#         left_synapses[i, 3:6] = np.random.rand(3)
#     elif 6 <= i < 9:
#         left_synapses[i, 6:9] = np.random.rand(3)
#     else:
#         left_synapses[i, 9:12] = np.random.rand(3)
# print(left_synapses)


right_synapses = np.random.rand(hidden_neurons_amount, output_neurons_amount)
print(right_synapses)

visible = input_sample
hidden = np.random.randint(0, 2, size=hidden_neurons_amount)
output = np.random.randint(0, 2, size=output_neurons_amount)

# -----------------------------
# אסטרטגיה (Strategy)
# -----------------------------
Temprature = 10
temp_decay = 0.95
iterations = 1000

print("Initial Visible State:", visible)
print("Initial Hidden State:", hidden)
print("Initial Output State:", output)
sys.exit(1)
for iteration in range(iterations):
    # Hidden Layer Update (Steps A-D)
    for j in range(hidden_neurons_amount):
        delta_E = hidden_bias[j] + np.sum(left_synapses[:, j] * visible)
    #    hidden[j] = 1 if np.random.rand() < sigmoid(delta_E / Temprature) else 0
        Pk = 1 / (1 + np.exp(-delta_E / Temprature))
        Xk = 1 if np.random.rand() < Pk else 0
        hidden[j] = Xk
    
    # for j in range(hidden_neurons_amount):
    #     Pk = 1 / (1 + np.exp(-delta_E / Temprature))
    #     hidden[j] = 1 if np.random.rand() < Pk else 0
    
    # Output Layer Update (Steps A-D)
    for k in range(output_neurons_amount):
        delta_E = output_bias[k] + np.sum(right_synapses[:, k] * hidden)
        # output[k] = 1 if np.random.rand() < sigmoid(delta_E / Temprature) else 0
    
    # for k in range(output_neurons_amount):
        Pk = 1 / (1 + np.exp(-delta_E / Temprature))
        Xk = 1 if np.random.rand() < Pk else 0
        output[k] = 1 if np.random.rand() < Xk else 0
    
    current_energy = energy(visible, hidden, output, visible_bias, hidden_bias, output_bias, left_synapses, right_synapses)
    print(f"Iteration {iteration + 1}, Energy: {current_energy}, Temperature: {Temprature}")
    Temprature *= temp_decay
    print("current Output State:", output)
    if Temprature < 0.01:
        break

# -----------------------------
# תחנה עצירה (Stop Condition)
# -----------------------------
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
