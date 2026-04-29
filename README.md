# First MLP (Framework)
- Single hidden layer, tanh activation functions
- Stochastic mini-batch gradient descent
- Normalizing helpers

## Layout
MLP Class with random initialization of weights:
  - forward_push()
  - backprop()
  - train()
Helpers:
  - BCE_Loss()
  - calculate_F1()
  - Parse_Data()
  - standardize()

## Usage
<python3 main.py <dataset> <hidden_units> <epochs> <learning_rate>
