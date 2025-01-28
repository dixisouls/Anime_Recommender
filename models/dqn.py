import torch
import torch.nn as nn
from typing import List


class DQNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int],
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the Deep Q-Network.

        Args:
            state_size (int): Dimension of the input state.
            action_size (int): Dimension of the action space.
            hidden_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(DQNetwork, self).__init__()

        # List to hold the layers of the network
        layers = []

        # Input layer
        layers.extend(
            [
                nn.Linear(state_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        )

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

        # Initialize weights of the network
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the network layers.

        Args:
            module: A module in the network.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size).
        """
        # Add batch dimension if missing
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.network(x)

    def save(self, path: str):
        """
        Save the model state to a file.

        Args:
            path (str): Path to save the model state.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        """
        Load the model state from a file.

        Args:
            path (str): Path to the model state file.
            device (torch.device): Device to map the model to.
        """
        self.load_state_dict(torch.load(path, map_location=device))
