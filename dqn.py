"""
Module for the Deep Q-Network (DQN) model.

This module provides the DQN class, which defines the architecture of a neural network used for Deep Q-Learning.

Classes:
    DQN: A class to define the neural network architecture for the DQN model.

Usage Example:
    model = DQN(in_states=3, h1_nodes=512, out_size=8)
    output = model(input_tensor)
"""

from torch import nn
from torch.nn import functional as F

class DQN(nn.Module):
    """
    A class to define the neural network architecture for the DQN model.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Fourth fully connected layer.
        out (nn.Linear): Output layer.

    Methods:
        forward(x): Perform a forward pass through the network.
    """

    def __init__(self, in_states, h1_nodes, out_size):
        """
        Initialize the DQN model with the given architecture.

        Parameters:
            in_states (int): The number of input states.
            h1_nodes (int): The number of nodes in the first hidden layer.
            out_size (int): The size of the output layer.
        """
        super().__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(in_states * 8 * 8, h1_nodes)   # First fully connected layer
        self.fc2 = nn.Linear(h1_nodes, 128)                 # Second fully connected layer
        self.fc3 = nn.Linear(128, 64)                       # Third fully connected layer
        self.fc4 = nn.Linear(64, 32)                        # Fourth fully connected layer
        self.out = nn.Linear(32, out_size * out_size)       # Output layer

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = x.view(x.size(0), -1)         # Flatten the input tensor
        x = F.relu(self.fc1(x))           # Apply ReLU activation to first layer
        x = F.relu(self.fc2(x))           # Apply ReLU activation to second layer
        x = F.relu(self.fc3(x))           # Apply ReLU activation to third layer
        x = F.relu(self.fc4(x))           # Apply ReLU activation to fourth layer
        x = self.out(x)                   # Calculate output
        
        # Reshape the output to [64, 64]
        x = x.view(-1, 64, 64)            # Reshape to [64, 64]
        
        # Squeeze to remove the batch dimension if batch_size is 1
        x = x.squeeze(0)                  # Remove the batch dimension
        
        return x
