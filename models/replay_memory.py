from collections import deque
import random
import torch
from typing import Tuple


class ReplayMemory:
    def __init__(self, capacity: int):
        """
        Initialize the replay memory with a given capacity.

        Args:
            capacity (int): Maximum number of transitions to store in memory.
        """
        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """
        Save a transition in the replay memory.

        Args:
            state (torch.Tensor): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (torch.Tensor): The next state.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions from the replay memory.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[torch.Tensor, ...]: Batch of sampled transitions.
        """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert lists of transitions into tensors
        return (
            torch.stack(state),
            torch.tensor(action),
            torch.tensor(reward),
            torch.stack(next_state),
            torch.tensor(done),
        )

    def __len__(self) -> int:
        """
        Return the current size of the replay memory.

        Returns:
            int: Number of transitions currently stored in memory.
        """
        return len(self.memory)
