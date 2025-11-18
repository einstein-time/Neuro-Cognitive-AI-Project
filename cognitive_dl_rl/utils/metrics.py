"""
Utility functions for computing performance metrics.
"""

import numpy as np
from typing import Tuple, List


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute accuracy.

    Args:
        predictions: Predicted values.
        targets: Target values.

    Returns:
        Accuracy score.
    """
    return np.mean(np.abs(predictions - targets) < 0.5)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted values.
        targets: Target values.

    Returns:
        MAE score.
    """
    return np.mean(np.abs(predictions - targets))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values.
        targets: Target values.

    Returns:
        RMSE score.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Pearson correlation.

    Args:
        predictions: Predicted values.
        targets: Target values.

    Returns:
        Correlation coefficient.
    """
    return np.corrcoef(predictions, targets)[0, 1]
