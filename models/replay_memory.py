from collections import deque
import random
import torch
from typing import Tuple


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.stack(state),
            torch.tensor(action),
            torch.tensor(reward),
            torch.stack(next_state),
            torch.tensor(done),
        )

    def __len__(self) -> int:
        return len(self.memory)
