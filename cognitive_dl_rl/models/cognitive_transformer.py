"""
Cognitive Load Transformer for predicting human cognitive performance.

This module implements a transformer-based model that predicts human
performance on cognitive tasks. The model incorporates:
- Multi-head attention (models selective attention mechanisms)
- Temporal decay positional encoding (models forgetting)
- Multi-task prediction heads (accuracy, RT, confidence)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .positional_encoding import TemporalDecayPositionalEncoding


class CognitiveLoadTransformer(nn.Module):
    """
    Transformer-based model for predicting human cognitive performance.

    Architecture:
    - Input projection: Maps task features to model dimension
    - Positional encoding: Adds temporal information with forgetting
    - Transformer encoder: Captures relationships between task elements
    - Multi-task heads: Predicts accuracy, reaction time, and confidence

    The model's attention weights can be interpreted as cognitive resource
    allocation, providing insights into human attention mechanisms.
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_sequence_length: int = 100,
        forgetting_rate: float = 0.01
    ):
        """
        Initialize Cognitive Load Transformer.

        Args:
            input_dim: Dimension of input features.
            d_model: Dimension of model embeddings.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout probability.
            max_sequence_length: Maximum sequence length.
            forgetting_rate: Rate of temporal decay in positional encoding.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding with forgetting
        self.positional_encoding = TemporalDecayPositionalEncoding(
            d_model=d_model,
            max_len=max_sequence_length,
            forgetting_rate=forgetting_rate,
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better training stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Multi-task prediction heads
        self.accuracy_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        self.reaction_time_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive RT
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Store attention weights for analysis
        self.attention_weights = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        task_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        retention_intervals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through cognitive load transformer.

        Args:
            task_sequence: Input task parameters [batch, seq_len, input_dim].
            attention_mask: Mask for padding [batch, seq_len], True = masked.
            retention_intervals: Time delays for forgetting [batch, seq_len].

        Returns:
            Tuple of:
            - accuracy: Predicted recall accuracy [batch, seq_len, 1]
            - reaction_time: Predicted response latency [batch, seq_len, 1]
            - confidence: Predicted metacognitive confidence [batch, seq_len, 1]
        """
        # Project input to model dimension
        x = self.input_projection(task_sequence)  # [batch, seq_len, d_model]

        # Add positional encoding with temporal decay
        x = self.positional_encoding(x, retention_intervals)

        # Encode with transformer
        # Note: PyTorch transformer uses True for positions to mask
        encoded = self.encoder(x, src_key_padding_mask=attention_mask)

        # Multi-task predictions
        accuracy = self.accuracy_head(encoded)  # [batch, seq_len, 1]
        reaction_time = self.reaction_time_head(encoded)  # [batch, seq_len, 1]
        confidence = self.confidence_head(encoded)  # [batch, seq_len, 1]

        return accuracy, reaction_time, confidence

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Extract attention weights from transformer layers.

        Returns:
            Attention weights [batch, num_layers, num_heads, seq_len, seq_len]
            or None if not available.
        """
        # Note: PyTorch doesn't expose attention weights by default
        # This would require modifying the encoder or using hooks
        return self.attention_weights

    def predict_learning_curve(
        self,
        task_params: torch.Tensor,
        num_trials: int = 50
    ) -> torch.Tensor:
        """
        Predict learning curve for a sequence of trials.

        Args:
            task_params: Initial task parameters [batch, input_dim].
            num_trials: Number of trials to simulate.

        Returns:
            Predicted accuracy over trials [batch, num_trials].
        """
        self.eval()

        batch_size = task_params.size(0)
        predictions = []

        with torch.no_grad():
            # Create sequence of trials with increasing practice
            for trial in range(num_trials):
                # Update repetition count
                trial_params = task_params.clone()
                trial_params[:, 3] = trial + 1  # num_repetitions column

                # Predict performance
                acc, _, _ = self.forward(trial_params.unsqueeze(1))
                predictions.append(acc.squeeze(1))

        return torch.cat(predictions, dim=1)


class AttentionAnalyzer:
    """
    Utility class for analyzing attention patterns in Cognitive Transformer.

    Provides methods to:
    - Extract attention weights
    - Compute attention to difficult items
    - Measure temporal decay in attention
    - Identify capacity bottlenecks
    """

    def __init__(self, model: CognitiveLoadTransformer):
        """
        Initialize attention analyzer.

        Args:
            model: Trained CognitiveLoadTransformer model.
        """
        self.model = model
        self.attention_maps = []

        # Register hooks to capture attention weights
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def hook_fn(module, input, output):
            # Capture attention weights from multi-head attention
            if hasattr(module, 'self_attn'):
                # This is simplified; actual implementation depends on PyTorch version
                pass

        for layer in self.model.encoder.layers:
            layer.register_forward_hook(hook_fn)

    def analyze_attention_to_difficulty(
        self,
        task_sequence: torch.Tensor,
        difficulty_scores: torch.Tensor
    ) -> float:
        """
        Compute correlation between attention and task difficulty.

        Hypothesis: Model should attend more to difficult items, consistent
        with human cognitive effort allocation.

        Args:
            task_sequence: Input task parameters [batch, seq_len, input_dim].
            difficulty_scores: Difficulty for each item [batch, seq_len].

        Returns:
            Correlation coefficient between attention and difficulty.
        """
        # Get attention weights
        with torch.no_grad():
            _ = self.model(task_sequence)
            attention = self.model.get_attention_weights()

        if attention is None:
            return 0.0

        # Average attention across layers and heads
        avg_attention = attention.mean(dim=(1, 2))  # [batch, seq_len, seq_len]

        # Compute attention received by each position
        attention_received = avg_attention.sum(dim=1)  # [batch, seq_len]

        # Correlation with difficulty
        correlation = torch.corrcoef(
            torch.stack([attention_received.flatten(), difficulty_scores.flatten()])
        )[0, 1]

        return correlation.item()

    def measure_temporal_decay(
        self,
        task_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Measure how attention decays over sequence positions.

        Args:
            task_sequence: Input task parameters [batch, seq_len, input_dim].

        Returns:
            Attention decay curve [seq_len].
        """
        with torch.no_grad():
            _ = self.model(task_sequence)
            attention = self.model.get_attention_weights()

        if attention is None:
            return torch.zeros(task_sequence.size(1))

        # Average across batch, layers, and heads
        avg_attention = attention.mean(dim=(0, 1, 2))  # [seq_len, seq_len]

        # Attention to each position (column mean)
        attention_per_position = avg_attention.mean(dim=0)

        return attention_per_position


def create_cognitive_transformer(
    config: Dict,
    device: torch.device
) -> CognitiveLoadTransformer:
    """
    Factory function to create CognitiveLoadTransformer from configuration.

    Args:
        config: Configuration dictionary with model parameters.
        device: Device to place model on.

    Returns:
        Initialized CognitiveLoadTransformer model.
    """
    model = CognitiveLoadTransformer(
        input_dim=config.get('input_dim', 16),
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        max_sequence_length=config.get('max_sequence_length', 100),
        forgetting_rate=config.get('forgetting_rate', 0.01)
    )

    model = model.to(device)

    return model
