import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Optional
import logging
from tqdm import tqdm
import numpy as np

from models.dqn import DQNetwork
from models.replay_memory import ReplayMemory

logger = logging.getLogger(__name__)


class DQLAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int],
        device: torch.device,
        batch_size: int = 64,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        target_update: int = 10,
        memory_capacity: int = 10000,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.0001,
    ):
        """
        Initialize the DQL Agent.

        Args:
            state_size (int): Dimension of the input state.
            action_size (int): Dimension of the action space.
            hidden_sizes (List[int]): List of hidden layer sizes.
            device (torch.device): Device to run the model on.
            batch_size (int): Size of each training batch.
            gamma (float): Discount factor for future rewards.
            eps_start (float): Initial epsilon value for epsilon-greedy policy.
            eps_end (float): Minimum epsilon value.
            eps_decay (float): Decay rate for epsilon.
            target_update (int): Frequency of target network updates.
            memory_capacity (int): Capacity of the replay memory.
            dropout_rate (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.epsilon = eps_start
        self.learning_rate = learning_rate

        # Initialize policy and target networks
        self.policy_net = DQNetwork(
            state_size, action_size, hidden_sizes, dropout_rate
        ).to(device)
        self.target_net = DQNetwork(
            state_size, action_size, hidden_sizes, dropout_rate
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay memory
        self.memory = ReplayMemory(memory_capacity)

        logger.info(
            f"DQL Agent initialized with state size {state_size} and action size {action_size}"
        )

    def select_action(
        self, state: torch.Tensor, valid_actions: Optional[torch.Tensor] = None
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (torch.Tensor): Current state tensor.
            valid_actions (Optional[torch.Tensor]): Mask of valid actions.

        Returns:
            int: Index of the selected action.
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)

                if valid_actions is not None:
                    # Ensure valid_actions has the same shape as q_values
                    valid_actions = valid_actions.to(self.device)
                    if len(valid_actions.shape) == 1:
                        valid_actions = valid_actions.unsqueeze(0)

                    # Apply mask by setting invalid action values to negative infinity
                    invalid_mask = ~valid_actions
                    q_values[invalid_mask] = float("-inf")

                return q_values.argmax(dim=1)[0].item()
        else:
            if valid_actions is not None:
                valid_indices = torch.where(valid_actions)[0]
                return random.choice(valid_indices.tolist())
            return random.randrange(self.action_size)

    def train(self, env, num_episodes: int) -> List[float]:
        """
        Train the agent.

        Args:
            env: The environment to train the agent in.
            num_episodes (int): Number of episodes to train for.

        Returns:
            List[float]: List of rewards per episode.
        """
        episode_rewards = []

        for episode in tqdm(range(num_episodes), desc="Training"):
            # Reset environment with random user
            user_id = random.choice(env.users)
            state = env.reset(user_id)
            episode_reward = 0

            while True:
                # Get valid actions (unwatched animes)
                valid_actions = env.get_valid_actions(state)

                # Select and perform action
                action = self.select_action(state, valid_actions)
                next_state, reward, done = env.step(action, user_id)
                episode_reward += reward

                # Store transition in replay memory
                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                # Train model if enough samples are available
                if len(self.memory) >= self.batch_size:
                    self._optimize_model()

                if done:
                    break

            # Update target network periodically
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay epsilon
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

            episode_rewards.append(episode_reward)

            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"Episode {episode + 1}, Average Reward: {avg_reward:.3f}, Epsilon: {self.epsilon:.3f}"
                )

        return episode_rewards

    def _optimize_model(self):
        """
        Perform one step of optimization on the policy network.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.memory.sample(self.batch_size)
        )

        # Move tensors to the appropriate device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Compute V(s_{t+1})
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[~done_batch] = self.target_net(
                next_state_batch[~done_batch]
            ).max(1)[0]

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save(self, path: str):
        """
        Save the agent's state to a file.

        Args:
            path (str): Path to save the agent's state.
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """
        Load the agent's state from a file.

        Args:
            path (str): Path to the file containing the agent's state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        logger.info(f"Agent loaded from {path}")
