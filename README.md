# Restricted Boltzmann Machine (RBM) for Iris Classification

This project implements a Restricted Boltzmann Machine (RBM) for classifying the Iris dataset. The implementation includes the training of an RBM using stochastic updates and visualization of the network structure.

## Features
- Uses the Iris dataset from scikit-learn
- Discretizes attributes for compatibility with RBM
- Implements both the forward pass and training phase
- Uses an energy-based stopping criterion
- Visualizes the RBM network structure

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy scikit-learn matplotlib
```

## Usage
Run the script to initialize and train the RBM model:
```bash
python main.py
```

### Parameters
- `iterations`: Number of training iterations
- `learning_rate`: Defines the step size for weight updates
- `hidden_neurons_amount`: Number of hidden layer neurons
- `output_neurons_amount`: Number of output neurons for classification
- `temp_decay`: Decay rate of temperature during training
- `energy_threshold`: Determines when to stop training

## Implementation Details
- **Data Preprocessing**: The Iris dataset attributes are discretized before feeding them into the RBM.
- **Training Process**:
  - Positive phase: Computes activation probabilities for hidden units
  - Negative phase: Reconstructs visible layer and updates weights
  - Weights are updated using Contrastive Divergence (CD-1)
- **Stopping Criteria**:
  - Early stopping if energy changes fall below a threshold
  - Stops if the neuron update rate is minimal
- **Visualization**: The `draw_rbm_network()` function generates a visual representation of the RBM at various training stages.

## Example Output
```
Iteration 100, Energy: -5.32
Predicted Iris Species: setosa
```

## File Structure
```
├── main.py  # Main training script
├── RBM_visualizer.py  # Visualization functions
├── utils.py  # Helper functions for initialization and energy calculations
├── README.md  # Project documentation
```

## Acknowledgments
- Based on the Restricted Boltzmann Machine concept
- Uses the Iris dataset from scikit-learn
