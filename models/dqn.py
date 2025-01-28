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
        Initialize Deep Q-Network

        Args:
            state_size (int): Size of input state
            action_size (int): Size of action space
            hidden_sizes (List[int]): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
        """
        super(DQNetwork, self).__init__()

        # Build layers dynamically
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

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size)
        """
        # Ensure input tensor has correct shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        return self.network(x)

    def save(self, path: str):
        """Save model state"""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        """Load model state"""
        self.load_state_dict(torch.load(path, map_location=device))
