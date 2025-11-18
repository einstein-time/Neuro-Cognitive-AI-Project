"""
Configuration module for Cognitive Load Deep Learning and Reinforcement Learning project.

This module contains all hyperparameters and configuration settings for both
the Deep Learning prediction model and the Reinforcement Learning validation system.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DataConfig:
    """Configuration for cognitive task data generation and loading."""

    n_participants: int = 10000
    sequence_length: int = 30
    min_sequence_length: int = 10
    max_sequence_length: int = 50
    n_symbols: int = 26
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

    # Cognitive parameters based on psychological research
    mean_capacity: float = 5.0  # Cowan (2001) working memory capacity
    std_capacity: float = 1.5
    base_forgetting_rate: float = 0.01  # Ebbinghaus decay constant
    learning_rate_human: float = 0.3  # Power law learning coefficient

    # Task difficulty parameters
    min_difficulty: float = 0.1
    max_difficulty: float = 0.9


@dataclass
class TransformerConfig:
    """Configuration for Cognitive Load Transformer model."""

    input_dim: int = 16
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_sequence_length: int = 100

    # Forgetting mechanism parameters
    forgetting_rate: float = 0.01
    use_temporal_decay: bool = True

    # Output heads
    predict_accuracy: bool = True
    predict_reaction_time: bool = True
    predict_confidence: bool = True


@dataclass
class TrainingConfig:
    """Configuration for Deep Learning model training."""

    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5

    # Loss weights for multi-task learning
    accuracy_weight: float = 1.0
    reaction_time_weight: float = 0.5
    confidence_weight: float = 0.3

    # Optimization
    gradient_clip_norm: float = 1.0
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Reproducibility
    seed: int = 42


@dataclass
class RLEnvironmentConfig:
    """Configuration for RL cognitive task environment."""

    sequence_length: int = 20
    memory_capacity: int = 5  # Miller's 7±2, we use 5 for difficulty
    forgetting_rate: float = 0.01
    attention_bottleneck: int = 1  # Can only process 1 item per timestep

    # Reward structure
    correct_recall_reward: float = 1.0
    incorrect_recall_penalty: float = -0.5
    time_penalty: float = -0.01

    # Task parameters
    n_symbols: int = 26  # A-Z
    min_sequence_length: int = 10
    max_sequence_length: int = 30

    # Difficulty curriculum
    use_curriculum: bool = True
    curriculum_stages: int = 5


@dataclass
class RLAgentConfig:
    """Configuration for RL agent with cognitive constraints."""

    memory_capacity: int = 5
    hidden_dim: int = 128
    num_layers: int = 2

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training parameters
    num_episodes: int = 10000
    max_steps_per_episode: int = 100
    update_frequency: int = 2048  # Steps before policy update
    num_epochs_per_update: int = 10
    batch_size: int = 64

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


@dataclass
class ValidationConfig:
    """Configuration for hypothesis validation and statistical testing."""

    # Significance levels
    alpha: float = 0.05

    # Validation thresholds
    min_correlation: float = 0.7
    max_decay_difference: float = 0.01
    max_capacity_difference: float = 1.5

    # Analysis parameters
    num_bootstrap_samples: int = 1000
    confidence_interval: float = 0.95

    # Visualization
    figure_dpi: int = 300
    save_figures: bool = True
    figure_format: str = "png"
    output_dir: str = "results"


@dataclass
class Config:
    """Master configuration combining all sub-configurations."""

    data: DataConfig = DataConfig()
    transformer: TransformerConfig = TransformerConfig()
    training: TrainingConfig = TrainingConfig()
    rl_env: RLEnvironmentConfig = RLEnvironmentConfig()
    rl_agent: RLAgentConfig = RLAgentConfig()
    validation: ValidationConfig = ValidationConfig()

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.data.train_split + self.data.val_split + self.data.test_split == 1.0
        assert self.data.mean_capacity > 0
        assert self.transformer.d_model % self.transformer.nhead == 0
        assert self.training.batch_size > 0
        assert 0 < self.training.learning_rate < 1
        assert self.rl_env.memory_capacity > 0
        assert self.rl_agent.memory_capacity == self.rl_env.memory_capacity


def get_default_config() -> Config:
    """
    Get default configuration for the project.

    Returns:
        Config: Default configuration object with all parameters.
    """
    return Config()


def save_config(config: Config, filepath: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration object to save.
        filepath: Path to save configuration file.
    """
    import json
    from dataclasses import asdict

    with open(filepath, 'w') as f:
        json.dump(asdict(config), f, indent=2)


def load_config(filepath: str) -> Config:
    """
    Load configuration from file.

    Args:
        filepath: Path to configuration file.

    Returns:
        Config: Loaded configuration object.
    """
    import json

    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return Config(
        data=DataConfig(**config_dict['data']),
        transformer=TransformerConfig(**config_dict['transformer']),
        training=TrainingConfig(**config_dict['training']),
        rl_env=RLEnvironmentConfig(**config_dict['rl_env']),
        rl_agent=RLAgentConfig(**config_dict['rl_agent']),
        validation=ValidationConfig(**config_dict['validation'])
    )
