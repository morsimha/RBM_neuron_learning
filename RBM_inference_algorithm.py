import numpy as np
from sklearn.datasets import load_iris

# -----------------------------
# תנאים מוקדמים (Preconditions)
# -----------------------------
# 1. הפעלת אלגוריתם למידת מכונת בולצמן באופן מיטבי.
# 2. שימוש בפרמטרים a, b, J המתקבלים מאלגוריתם הלמידה.

# Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

visible_neurons_amount = input_sample.shape[0]
hidden_neurons_amount = 3
output_neurons_amount = 3

visible_bias = np.random.rand(visible_neurons_amount)
hidden_bias = np.random.rand(hidden_neurons_amount)
output_bias = np.random.rand(output_neurons_amount)

left_synapses = np.random.rand(visible_neurons_amount, hidden_neurons_amount)
right_synapses = np.random.rand(hidden_neurons_amount, output_neurons_amount)

visible = input_sample
hidden = np.random.randint(0, 2, size=hidden_neurons_amount)
output = np.random.randint(0, 2, size=output_neurons_amount)

# -----------------------------
# אסטרטגיה (Strategy)
# -----------------------------
Temprature = 1.0
temp_decay = 0.95
iterations = 1000

rng = np.random.default_rng()

for iteration in range(iterations):
    # Hidden Layer Update (Random State)
    hidden = rng.integers(0, 2, size=hidden_neurons_amount)
    
    # Output Layer Update (Random State)
    output = rng.integers(0, 2, size=output_neurons_amount)
    
    current_energy = energy(visible, hidden, output, visible_bias, hidden_bias, output_bias, left_synapses, right_synapses)
    print(f"Iteration {iteration + 1}, Energy: {current_energy}, Temperature: {Temprature}")
    Temprature *= temp_decay
    print("Current Output State:", output)
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
