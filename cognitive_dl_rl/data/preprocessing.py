"""
Data preprocessing utilities for cognitive task data.

This module handles:
- Data normalization and standardization
- Sequence padding for variable-length tasks
- Train/val/test splitting
- PyTorch Dataset and DataLoader creation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler


class CognitiveTaskDataset(Dataset):
    """
    PyTorch Dataset for cognitive task sequences.

    Handles variable-length sequences with padding and creates
    appropriate attention masks for transformer models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 30,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data: DataFrame with cognitive task data.
            sequence_length: Maximum sequence length for padding.
            feature_columns: Columns to use as input features.
            target_columns: Columns to use as targets.
            scaler: Pre-fitted scaler for features (optional).
            fit_scaler: Whether to fit scaler on this data.
        """
        self.data = data
        self.sequence_length = sequence_length

        if feature_columns is None:
            self.feature_columns = [
                'sequence_length', 'study_time', 'retention_interval',
                'num_repetitions', 'spacing', 'task_complexity',
                'cognitive_load', 'capacity', 'forgetting_rate'
            ]
        else:
            self.feature_columns = feature_columns

        if target_columns is None:
            self.target_columns = ['performance', 'reaction_time', 'confidence']
        else:
            self.target_columns = target_columns

        # Normalize features
        self.scaler = scaler
        if fit_scaler or scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(
                data[self.feature_columns].values
            )
        else:
            self.features = self.scaler.transform(
                data[self.feature_columns].values
            )

        self.targets = data[self.target_columns].values

        # Group by participant to create sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        """
        Create sequences grouped by participant.

        Returns:
            List of sequence dictionaries with features and targets.
        """
        sequences = []

        for pid in self.data['participant_id'].unique():
            participant_data = self.data[self.data['participant_id'] == pid]
            indices = participant_data.index.values

            # Get features and targets for this participant
            features = self.features[indices]
            targets = self.targets[indices]

            # Pad or truncate to sequence_length
            seq_len = min(len(features), self.sequence_length)

            padded_features = np.zeros((self.sequence_length, features.shape[1]))
            padded_targets = np.zeros((self.sequence_length, targets.shape[1]))
            attention_mask = np.zeros(self.sequence_length, dtype=bool)

            padded_features[:seq_len] = features[:seq_len]
            padded_targets[:seq_len] = targets[:seq_len]
            attention_mask[:seq_len] = True

            sequences.append({
                'features': padded_features,
                'targets': padded_targets,
                'attention_mask': attention_mask,
                'sequence_length': seq_len
            })

        return sequences

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sequence by index.

        Args:
            idx: Sequence index.

        Returns:
            Dictionary with tensors:
            - task_sequence: [seq_len, n_features]
            - target_accuracy: [seq_len, 1]
            - target_rt: [seq_len, 1]
            - target_confidence: [seq_len, 1]
            - attention_mask: [seq_len]
        """
        seq = self.sequences[idx]

        return {
            'task_sequence': torch.FloatTensor(seq['features']),
            'target_accuracy': torch.FloatTensor(seq['targets'][:, 0:1]),
            'target_rt': torch.FloatTensor(seq['targets'][:, 1:2]),
            'target_confidence': torch.FloatTensor(seq['targets'][:, 2:3]),
            'attention_mask': torch.BoolTensor(~seq['attention_mask']),  # Inverted for transformer
            'sequence_length': seq['sequence_length']
        }


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets by participant.

    Ensures that all trials from a participant stay in the same split.

    Args:
        data: Full dataset.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Get unique participants
    participants = data['participant_id'].unique()
    n_participants = len(participants)

    # Shuffle participants
    rng = np.random.RandomState(seed)
    rng.shuffle(participants)

    # Split participants
    train_end = int(n_participants * train_ratio)
    val_end = int(n_participants * (train_ratio + val_ratio))

    train_pids = participants[:train_end]
    val_pids = participants[train_end:val_end]
    test_pids = participants[val_end:]

    # Split data
    train_data = data[data['participant_id'].isin(train_pids)].reset_index(drop=True)
    val_data = data[data['participant_id'].isin(val_pids)].reset_index(drop=True)
    test_data = data[data['participant_id'].isin(test_pids)].reset_index(drop=True)

    return train_data, val_data, test_data


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    batch_size: int = 32,
    sequence_length: int = 30,
    num_workers: int = 4,
    feature_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_data: Training data.
        val_data: Validation data.
        test_data: Test data.
        batch_size: Batch size for training.
        sequence_length: Maximum sequence length.
        num_workers: Number of data loading workers.
        feature_columns: Feature column names.
        target_columns: Target column names.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler).
    """
    # Create datasets (fit scaler on train, apply to val/test)
    train_dataset = CognitiveTaskDataset(
        train_data,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
        target_columns=target_columns,
        fit_scaler=True
    )

    val_dataset = CognitiveTaskDataset(
        val_data,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    test_dataset = CognitiveTaskDataset(
        test_data,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, train_dataset.scaler


def normalize_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    scaler: Optional[StandardScaler] = None,
    fit: bool = False
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.

    Args:
        data: DataFrame with features.
        feature_columns: Columns to normalize.
        scaler: Pre-fitted scaler (optional).
        fit: Whether to fit scaler on this data.

    Returns:
        Tuple of (normalized_features, scaler).
    """
    features = data[feature_columns].values

    if fit or scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
    else:
        normalized = scaler.transform(features)

    return normalized, scaler
