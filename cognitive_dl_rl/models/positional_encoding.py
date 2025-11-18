"""
Temporal positional encoding with forgetting mechanism for cognitive modeling.

This module implements a custom positional encoding that incorporates
the Ebbinghaus forgetting curve, allowing the model to capture memory
decay over time in sequential learning tasks.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class TemporalDecayPositionalEncoding(nn.Module):
    """
    Positional encoding modified to incorporate forgetting over time.

    Standard positional encoding adds sinusoidal patterns to capture position.
    We extend this to model the Ebbinghaus forgetting curve: R(t) = exp(-λt),
    where t is the retention interval and λ is the forgetting rate.

    This allows the transformer to learn that recent information is more
    reliable than older information, consistent with human memory.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        forgetting_rate: float = 0.01,
        dropout: float = 0.1
    ):
        """
        Initialize temporal decay positional encoding.

        Args:
            d_model: Dimension of model embeddings.
            max_len: Maximum sequence length.
            forgetting_rate: Decay constant for forgetting curve.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.forgetting_rate = forgetting_rate
        self.dropout = nn.Dropout(p=dropout)

        # Create standard positional encoding
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(
        self,
        x: torch.Tensor,
        retention_intervals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply positional encoding with optional forgetting decay.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            retention_intervals: Time delays for each position [batch, seq_len].
                If None, uses positional indices as time proxy.

        Returns:
            Tensor with positional encoding and forgetting applied [batch, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.size()

        # Add standard positional encoding
        x = x + self.pe[:, :seq_len, :]

        # Apply forgetting decay if retention intervals provided
        if retention_intervals is not None:
            # Compute decay: exp(-λt)
            decay = torch.exp(-self.forgetting_rate * retention_intervals)
            decay = decay.unsqueeze(-1)  # [batch, seq_len, 1]

            # Apply decay to encoded representation
            x = x * decay

        return self.dropout(x)


class LearnableTemporalEncoding(nn.Module):
    """
    Learnable temporal encoding that adapts forgetting rate during training.

    Unlike fixed exponential decay, this learns optimal decay patterns
    from data, allowing for more flexible modeling of memory dynamics.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        dropout: float = 0.1
    ):
        """
        Initialize learnable temporal encoding.

        Args:
            d_model: Dimension of model embeddings.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # Learnable decay function (MLP)
        self.decay_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        retention_intervals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply learnable temporal encoding.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            retention_intervals: Time delays [batch, seq_len] (optional).

        Returns:
            Encoded tensor [batch, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.size()

        # Add learnable position embeddings
        positions = torch.arange(seq_len, device=x.device)
        position_embeds = self.position_embeddings(positions)  # [seq_len, d_model]
        x = x + position_embeds.unsqueeze(0)  # Broadcast to batch

        # Apply learned decay if retention intervals provided
        if retention_intervals is not None:
            # Normalize intervals for numerical stability
            normalized_intervals = retention_intervals / (retention_intervals.max() + 1e-8)
            normalized_intervals = normalized_intervals.unsqueeze(-1)  # [batch, seq_len, 1]

            # Compute learned decay
            decay = self.decay_network(normalized_intervals)  # [batch, seq_len, 1]

            # Apply decay
            x = x * decay

        return self.dropout(x)


class CognitiveLoadEncoding(nn.Module):
    """
    Encoding that incorporates both temporal decay and cognitive load effects.

    Combines:
    - Positional information (sequence position)
    - Temporal decay (forgetting over time)
    - Cognitive load (capacity constraints affect encoding)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        forgetting_rate: float = 0.01,
        dropout: float = 0.1
    ):
        """
        Initialize cognitive load encoding.

        Args:
            d_model: Dimension of model embeddings.
            max_len: Maximum sequence length.
            forgetting_rate: Forgetting decay rate.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.forgetting_rate = forgetting_rate
        self.dropout = nn.Dropout(p=dropout)

        # Standard positional encoding
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2))
        )

        pe = torch.zeros(max_len, d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model // 2 > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        self.register_buffer('pe', pe.unsqueeze(0))

        # Cognitive load embedding
        self.load_embedding = nn.Linear(1, d_model // 2)

    def forward(
        self,
        x: torch.Tensor,
        retention_intervals: Optional[torch.Tensor] = None,
        cognitive_load: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cognitive load-aware encoding.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            retention_intervals: Time delays [batch, seq_len].
            cognitive_load: Load values [batch, seq_len].

        Returns:
            Encoded tensor [batch, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.size()

        # Split into position and load components
        d_half = self.d_model // 2

        # Add positional encoding to first half
        x_pos = x[:, :, :d_half] + self.pe[:, :seq_len, :]

        # Add cognitive load encoding to second half
        if cognitive_load is not None:
            load_embed = self.load_embedding(cognitive_load.unsqueeze(-1))  # [batch, seq_len, d_half]
            x_load = x[:, :, d_half:] + load_embed
        else:
            x_load = x[:, :, d_half:]

        # Concatenate
        x = torch.cat([x_pos, x_load], dim=-1)

        # Apply forgetting decay
        if retention_intervals is not None:
            decay = torch.exp(-self.forgetting_rate * retention_intervals)
            decay = decay.unsqueeze(-1)
            x = x * decay

        return self.dropout(x)


def create_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_len: int = 100,
    forgetting_rate: float = 0.01,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function to create positional encoding module.

    Args:
        encoding_type: Type of encoding ('temporal_decay', 'learnable', 'cognitive_load').
        d_model: Model dimension.
        max_len: Maximum sequence length.
        forgetting_rate: Forgetting rate for temporal decay.
        dropout: Dropout probability.

    Returns:
        Positional encoding module.
    """
    if encoding_type == 'temporal_decay':
        return TemporalDecayPositionalEncoding(d_model, max_len, forgetting_rate, dropout)
    elif encoding_type == 'learnable':
        return LearnableTemporalEncoding(d_model, max_len, dropout)
    elif encoding_type == 'cognitive_load':
        return CognitiveLoadEncoding(d_model, max_len, forgetting_rate, dropout)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
