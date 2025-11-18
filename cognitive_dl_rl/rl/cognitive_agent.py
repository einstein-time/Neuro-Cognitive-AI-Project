"""
RL agent with human-like cognitive architecture.

Implements a PPO-based agent that learns under cognitive constraints:
- Limited memory capacity
- Attention as policy (resource allocation)
- Learns to space practice optimally
- Develops human-like learning patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import deque


class CognitiveAgent(nn.Module):
    """
    RL agent with cognitive architecture for memory tasks.

    Components:
    - Working memory encoder (processes memory state)
    - Attention controller (policy network - decides what to rehearse)
    - Value network (metacognitive monitor - estimates expected performance)
    - Memory management (learns optimal forgetting/retention strategies)
    """

    def __init__(
        self,
        n_symbols: int = 26,
        memory_capacity: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        """
        Initialize cognitive agent.

        Args:
            n_symbols: Number of unique symbols in environment.
            memory_capacity: Working memory capacity (number of slots).
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
        """
        super().__init__()

        self.n_symbols = n_symbols
        self.memory_capacity = memory_capacity
        self.hidden_dim = hidden_dim

        # Input dimension: symbol (one-hot) + memory state + phase + load
        input_dim = n_symbols + memory_capacity + 1 + 1

        # Shared feature extractor
        layers = []
        prev_dim = input_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Policy head (attention controller)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, memory_capacity + 1)  # +1 for skip action
        )

        # Value head (metacognitive monitor)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Initialize policy head with smaller weights for exploration
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_head[-1].bias, 0.0)

    def forward(self, observation: Dict[str, torch.Tensor]) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass through agent network.

        Args:
            observation: Dictionary with:
                - symbol: [batch, n_symbols]
                - memory_state: [batch, memory_capacity]
                - phase: [batch] (scalar)
                - load: [batch, 1]

        Returns:
            Tuple of (action_distribution, value_estimate).
        """
        # Concatenate all observation components
        symbol = observation['symbol']
        memory_state = observation['memory_state']
        phase = observation['phase'].unsqueeze(-1).float()
        load = observation['load']

        state = torch.cat([symbol, memory_state, phase, load], dim=-1)

        # Extract features
        features = self.feature_extractor(state)

        # Policy (attention allocation)
        policy_logits = self.policy_head(features)
        action_dist = Categorical(logits=policy_logits)

        # Value (expected return)
        value = self.value_head(features)

        return action_dist, value

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given observation.

        Args:
            observation: Environment observation.
            deterministic: If True, select greedy action.

        Returns:
            Tuple of (action, log_prob, value).
        """
        # Convert observation to tensors
        obs_tensor = self._obs_to_tensor(observation)

        with torch.no_grad():
            action_dist, value = self.forward(obs_tensor)

            if deterministic:
                action = action_dist.probs.argmax()
            else:
                action = action_dist.sample()

            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations (used in PPO update).

        Args:
            observations: Batch of observations.
            actions: Batch of actions taken.

        Returns:
            Tuple of (action_log_probs, values, entropy).
        """
        action_dist, values = self.forward(observations)

        action_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return action_log_probs, values.squeeze(-1), entropy

    def _obs_to_tensor(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy observation to tensor."""
        obs_tensor = {}

        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                tensor = torch.FloatTensor(value)
            else:
                tensor = torch.tensor([value], dtype=torch.float32)

            # Add batch dimension if needed
            if tensor.ndim == 1 and key != 'phase':
                tensor = tensor.unsqueeze(0)
            elif key == 'phase':
                tensor = tensor.unsqueeze(0) if tensor.ndim == 0 else tensor

            obs_tensor[key] = tensor

        return obs_tensor


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for cognitive agent.

    Implements PPO algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function loss
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        agent: CognitiveAgent,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize PPO trainer.

        Args:
            agent: CognitiveAgent to train.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            clip_epsilon: PPO clipping parameter.
            value_loss_coef: Coefficient for value loss.
            entropy_coef: Coefficient for entropy bonus.
            max_grad_norm: Maximum gradient norm for clipping.
            device: Device for training.
        """
        self.agent = agent.to(device)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Array of rewards [num_steps].
            values: Array of value estimates [num_steps].
            dones: Array of done flags [num_steps].

        Returns:
            Tuple of (advantages, returns).
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(
        self,
        rollout_buffer: Dict[str, np.ndarray],
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update agent using PPO algorithm.

        Args:
            rollout_buffer: Dictionary with trajectory data.
            num_epochs: Number of optimization epochs.
            batch_size: Minibatch size.

        Returns:
            Dictionary of training metrics.
        """
        # Extract rollout data
        observations = rollout_buffer['observations']
        actions = rollout_buffer['actions']
        old_log_probs = rollout_buffer['log_probs']
        rewards = rollout_buffer['rewards']
        values = rollout_buffer['values']
        dones = rollout_buffer['dones']

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        # Convert observations
        obs_tensors = self._convert_obs_batch(observations)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # Multiple epochs of optimization
        for _ in range(num_epochs):
            # Random minibatches
            indices = np.arange(len(actions))
            np.random.shuffle(indices)

            for start in range(0, len(actions), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_obs = {
                    key: value[batch_indices] for key, value in obs_tensors.items()
                }
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate actions
                new_log_probs, new_values, entropy = self.agent.evaluate_actions(
                    batch_obs, batch_actions
                )

                # PPO policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

        return metrics

    def _convert_obs_batch(self, observations: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert list of observations to batched tensors."""
        batch_obs = {}

        # Stack each component
        batch_obs['symbol'] = torch.FloatTensor(
            np.stack([obs['symbol'] for obs in observations])
        ).to(self.device)

        batch_obs['memory_state'] = torch.FloatTensor(
            np.stack([obs['memory_state'] for obs in observations])
        ).to(self.device)

        batch_obs['phase'] = torch.LongTensor(
            [obs['phase'] for obs in observations]
        ).to(self.device)

        batch_obs['load'] = torch.FloatTensor(
            np.stack([obs['load'] for obs in observations])
        ).to(self.device)

        return batch_obs
