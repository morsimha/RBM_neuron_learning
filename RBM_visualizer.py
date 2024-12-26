import numpy as np
import matplotlib.pyplot as plt

def draw_rbm_network(visible_state, hidden_state, output_state, left_synapses, right_synapses):
    # Network parameters
    num_visible = len(visible_state)
    num_hidden = len(hidden_state)
    num_output = len(output_state)

    # Node positions
    visible_y = np.linspace(0, 1, num_visible)
    hidden_y = np.linspace(0, 1, num_hidden)
    output_y = np.linspace(0, 1, num_output)

    visible_x = 0
    hidden_x = 0.5
    output_x = 1

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')

    # Draw visible, hidden, and output nodes
    for i, v in enumerate(visible_state):
        color = 'black' if v == 1 else 'white'
        circle = plt.Circle((visible_x, visible_y[i]), 0.03, color=color, ec='black')
        ax.add_artist(circle)

    for i, h in enumerate(hidden_state):
        color = 'black' if h == 1 else 'white'
        circle = plt.Circle((hidden_x, hidden_y[i]), 0.03, color=color, ec='black')
        ax.add_artist(circle)

    for i, o in enumerate(output_state):
        color = 'black' if o == 1 else 'white'
        circle = plt.Circle((output_x, output_y[i]), 0.03, color=color, ec='black')
        ax.add_artist(circle)

    # Draw left synapses and weights
    for i in range(num_visible):
        for j in range(num_hidden):
            x_coords = [visible_x, hidden_x]
            y_coords = [visible_y[i], hidden_y[j]]
            ax.plot(x_coords, y_coords, 'gray', lw=0.5)
            weight = left_synapses[i, j]
            mid_x = (visible_x + hidden_x) / 2
            mid_y = (visible_y[i] + hidden_y[j]) / 2
            ax.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8, color='blue')

    # Draw right synapses and weights
    for i in range(num_hidden):
        for j in range(num_output):
            x_coords = [hidden_x, output_x]
            y_coords = [hidden_y[i], output_y[j]]
            ax.plot(x_coords, y_coords, 'gray', lw=0.5)
            weight = right_synapses[i, j]
            mid_x = (hidden_x + output_x) / 2
            mid_y = (hidden_y[i] + output_y[j]) / 2
            ax.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8, color='green')

    plt.show()

# Input data
visible_state = np.array([0., 1., 1., 0., 0., 0., 0., 0.])
hidden_state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
output_state = np.array([1, 1, 1])

left_synapses = np.array([
    [0.79436224, 0.48648836, 0.85495067, 0.97060964, 0.2026345,  0.10018927,
     0.83278838, 0.77772938, 0.04003811, 0.25640702, 0.99357768, 0.65283307],
    [0.59583556, 0.70984534, 0.68084456, 0.39285446, 0.47066568, 0.05125829,
     0.14749417, 0.22798163, 0.44053466, 0.03266189, 0.63552243, 0.78599645],
    [0.21726649, 0.43121419, 0.1301434,  0.44362525, 0.30264522, 0.73570908,
     0.96016277, 0.84560592, 0.20980441, 0.60373767, 0.91091587, 0.85598155],
    [0.23458725, 0.95135864, 0.55978008, 0.19409072, 0.00262227, 0.8900581,
     0.11900772, 0.28331631, 0.01233025, 0.62468579, 0.30507394, 0.80658157],
    [0.04969197, 0.1088581,  0.85648073, 0.4565387,  0.38585067, 0.69885253,
     0.1309755,  0.46301329, 0.78750698, 0.71476334, 0.72121594, 0.8344664 ],
    [0.30034788, 0.92025587, 0.17759989, 0.29882182, 0.04246413, 0.69426632,
     0.10454134, 0.65837841, 0.06431199, 0.69460719, 0.61446837, 0.85651332],
    [0.38218818, 0.64130918, 0.79759798, 0.18577285, 0.14021215, 0.33261719,
     0.9489593,  0.0397333,  0.14584243, 0.22926158, 0.89183425, 0.90289173],
    [0.9012017,  0.7231504,  0.70302784, 0.237009,   0.66894617, 0.92978878,
     0.82550363, 0.13992375, 0.74829416, 0.43541464, 0.74467245, 0.82887055]
])

right_synapses = np.array([
    [0.40091533, 0.22761745, 0.17022489],
    [0.84796104, 0.67524663, 0.83218976],
    [0.25612344, 0.44487559, 0.57345343],
    [0.06945708, 0.16745691, 0.00219215],
    [0.7993215,  0.94164056, 0.30000315],
    [0.18746306, 0.03573912, 0.59452197],
    [0.33637505, 0.8551409,  0.01570168],
    [0.88882316, 0.4328441,  0.32128215],
    [0.31170599, 0.50424589, 0.06469053],
    [0.78819187, 0.54383689, 0.71413088],
    [0.77366005, 0.62856029, 0.94718298],
    [0.50091026, 0.28048902, 0.99360401]
])

# # Draw the RBM network
# draw_rbm_network(visible_state, hidden_state, output_state, left_synapses, right_synapses)
