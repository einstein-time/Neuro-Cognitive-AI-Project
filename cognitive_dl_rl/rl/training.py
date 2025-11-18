"""
Training loop for RL agent in cognitive environment.

Implements:
- Episode collection with rollout buffer
- PPO training updates
- Performance tracking and logging
- Learning curve analysis
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
from pathlib import Path

from .cognitive_environment import CognitiveTaskEnvironment, CurriculumCognitiveEnvironment
from .cognitive_agent import CognitiveAgent, PPOTrainer


class RolloutBuffer:
    """
    Buffer for storing episode trajectories.

    Stores observations, actions, rewards, values, and log probabilities
    for PPO training.
    """

    def __init__(self, buffer_size: int = 2048):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum number of steps to store.
        """
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.size = 0

    def add(
        self,
        observation: Dict,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add transition to buffer."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.size += 1

    def get(self) -> Dict[str, np.ndarray]:
        """
        Get buffer contents as numpy arrays.

        Returns:
            Dictionary with trajectory data.
        """
        return {
            'observations': self.observations,
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.buffer_size


def train_rl_agent(
    env: CognitiveTaskEnvironment,
    agent: CognitiveAgent,
    trainer: PPOTrainer,
    num_episodes: int = 10000,
    update_frequency: int = 2048,
    num_epochs_per_update: int = 10,
    batch_size: int = 64,
    eval_frequency: int = 100,
    save_frequency: int = 500,
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train RL agent in cognitive environment.

    Args:
        env: Cognitive task environment.
        agent: Cognitive agent to train.
        trainer: PPO trainer.
        num_episodes: Number of training episodes.
        update_frequency: Steps between PPO updates.
        num_epochs_per_update: Optimization epochs per update.
        batch_size: Minibatch size for PPO.
        eval_frequency: Episodes between evaluations.
        save_frequency: Episodes between checkpoints.
        checkpoint_dir: Directory for saving checkpoints.
        verbose: Whether to print progress.

    Returns:
        Training history dictionary.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_accuracies': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }

    # Rollout buffer
    buffer = RolloutBuffer(buffer_size=update_frequency)

    # Running statistics
    recent_rewards = deque(maxlen=100)
    recent_accuracies = deque(maxlen=100)

    # Training loop
    total_steps = 0
    num_updates = 0

    if verbose:
        print("Starting RL agent training...")
        print(f"Environment: {env.__class__.__name__}")
        print(f"Agent: {agent.__class__.__name__}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Update frequency: {update_frequency}")
        print("-" * 60)

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action, log_prob, value = agent.select_action(observation)

            # Environment step
            next_observation, reward, done, info = env.step(action)

            # Store transition
            buffer.add(observation, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            observation = next_observation

            # Update policy if buffer is full
            if buffer.is_full():
                rollout_data = buffer.get()
                metrics = trainer.update(
                    rollout_data,
                    num_epochs=num_epochs_per_update,
                    batch_size=batch_size
                )

                # Store metrics
                history['policy_losses'].append(metrics['policy_loss'])
                history['value_losses'].append(metrics['value_loss'])
                history['entropies'].append(metrics['entropy'])

                buffer.reset()
                num_updates += 1

        # Episode complete
        episode_accuracy = info.get('accuracy', 0.0)

        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['episode_accuracies'].append(episode_accuracy)

        recent_rewards.append(episode_reward)
        recent_accuracies.append(episode_accuracy)

        # Logging
        if verbose and (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(recent_rewards)
            avg_accuracy = np.mean(recent_accuracies)
            avg_policy_loss = np.mean(history['policy_losses'][-10:]) if history['policy_losses'] else 0
            avg_value_loss = np.mean(history['value_losses'][-10:]) if history['value_losses'] else 0

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Accuracy (last 100): {avg_accuracy:.3f}")
            print(f"  Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Total Steps: {total_steps}")
            print(f"  Updates: {num_updates}")
            print("-" * 60)

        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_file = checkpoint_path / f"agent_episode_{episode+1}.pth"
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'history': history
            }, checkpoint_file)

            if verbose:
                print(f"Saved checkpoint to {checkpoint_file}")

    # Save final model
    final_checkpoint = checkpoint_path / "agent_final.pth"
    torch.save({
        'episode': num_episodes,
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': history
    }, final_checkpoint)

    if verbose:
        print("\nTraining completed!")
        print(f"Final average reward: {np.mean(recent_rewards):.2f}")
        print(f"Final average accuracy: {np.mean(recent_accuracies):.3f}")

    return history


def evaluate_agent(
    env: CognitiveTaskEnvironment,
    agent: CognitiveAgent,
    num_episodes: int = 100,
    deterministic: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate trained agent.

    Args:
        env: Cognitive environment.
        agent: Trained agent.
        num_episodes: Number of evaluation episodes.
        deterministic: Use deterministic policy.
        verbose: Print progress.

    Returns:
        Dictionary of evaluation metrics.
    """
    agent.eval()

    episode_rewards = []
    episode_accuracies = []
    episode_lengths = []

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(observation, deterministic=deterministic)
            observation, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_accuracies.append(info.get('accuracy', 0.0))
        episode_lengths.append(episode_length)

        if verbose and (episode + 1) % 20 == 0:
            print(f"Evaluation episode {episode + 1}/{num_episodes}")

    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_accuracy': np.mean(episode_accuracies),
        'std_accuracy': np.std(episode_accuracies),
        'mean_length': np.mean(episode_lengths)
    }

    return metrics


def extract_learning_curve(history: Dict[str, List[float]], window: int = 100) -> np.ndarray:
    """
    Extract smoothed learning curve from training history.

    Args:
        history: Training history dictionary.
        window: Window size for moving average.

    Returns:
        Smoothed accuracy curve.
    """
    accuracies = np.array(history['episode_accuracies'])

    # Compute moving average
    smoothed = []
    for i in range(len(accuracies)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(accuracies[start:i+1]))

    return np.array(smoothed)


def analyze_spacing_patterns(
    env: CognitiveTaskEnvironment,
    agent: CognitiveAgent,
    num_episodes: int = 50
) -> Dict[str, float]:
    """
    Analyze whether agent learns spacing effect.

    Measures if agent tends to space practice (distributed) rather than
    mass practice (repeated attention to same item).

    Args:
        env: Cognitive environment.
        agent: Trained agent.
        num_episodes: Number of episodes to analyze.

    Returns:
        Dictionary with spacing metrics.
    """
    agent.eval()

    attention_patterns = []

    for _ in range(num_episodes):
        observation = env.reset()
        episode_actions = []
        done = False

        while not done:
            action, _, _ = agent.select_action(observation, deterministic=True)
            observation, _, done, _ = env.step(action)
            episode_actions.append(action)

        attention_patterns.append(episode_actions)

    # Analyze patterns
    # Look for repeated vs. distributed attention
    spacing_scores = []

    for actions in attention_patterns:
        # Compute how distributed attention is
        unique_actions = len(set(actions))
        max_possible = env.memory_capacity + 1
        spacing_score = unique_actions / max_possible
        spacing_scores.append(spacing_score)

    return {
        'mean_spacing_score': np.mean(spacing_scores),
        'std_spacing_score': np.std(spacing_scores)
    }
