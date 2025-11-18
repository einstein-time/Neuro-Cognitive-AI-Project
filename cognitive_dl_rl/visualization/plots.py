"""
Visualization functions for cognitive modeling project.

Provides plots for:
- Learning curves
- Forgetting curves
- Training progress
- Comparative analysis
- Attention patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_learning_curves(
    human_curve: np.ndarray,
    rl_curve: np.ndarray,
    dl_curve: Optional[np.ndarray] = None,
    title: str = "Learning Curves: Human vs RL Agent",
    save_path: Optional[str] = None
):
    """
    Plot learning curves comparison.

    Args:
        human_curve: Human performance curve.
        rl_curve: RL agent performance curve.
        dl_curve: DL model predictions (optional).
        title: Plot title.
        save_path: Path to save figure (optional).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    trials = np.arange(len(human_curve))

    ax.plot(trials, human_curve, 'o-', label='Human', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(trials, rl_curve, 's-', label='RL Agent', linewidth=2, markersize=4, alpha=0.7)

    if dl_curve is not None:
        dl_trials = np.arange(len(dl_curve))
        ax.plot(dl_trials, dl_curve, '^-', label='DL Prediction', linewidth=2, markersize=4, alpha=0.7)

    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Recall Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_forgetting_curves(
    human_curve: Dict[float, float],
    rl_curve: Dict[float, float],
    title: str = "Forgetting Curves: Ebbinghaus Decay",
    save_path: Optional[str] = None
):
    """
    Plot forgetting curves comparison.

    Args:
        human_curve: Dict mapping retention interval to performance.
        rl_curve: Dict mapping retention interval to performance.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get common intervals
    all_intervals = sorted(set(list(human_curve.keys()) + list(rl_curve.keys())))

    human_retention = [human_curve.get(t, np.nan) for t in all_intervals]
    rl_retention = [rl_curve.get(t, np.nan) for t in all_intervals]

    ax.plot(all_intervals, human_retention, 'o-', label='Human', linewidth=2, markersize=6)
    ax.plot(all_intervals, rl_retention, 's-', label='RL Agent', linewidth=2, markersize=6)

    ax.set_xlabel('Retention Interval (seconds)', fontsize=12)
    ax.set_ylabel('Recall Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Log scale for time
    ax.set_xscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_progress(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None
):
    """
    Plot training metrics over time.

    Args:
        history: Dictionary with training history.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Correlation
    if 'val_correlation' in history:
        axes[0, 1].plot(history['val_correlation'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].set_title('Validation Correlation')
        axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # RL rewards
    if 'episode_rewards' in history:
        # Smooth rewards
        window = 100
        rewards = np.array(history['episode_rewards'])
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(smoothed)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward (smoothed)')
        axes[1, 1].set_title('RL Training Rewards')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_comparative_analysis(
    validation_results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot comprehensive comparative analysis.

    Args:
        validation_results: Results from hypothesis validation.
        save_path: Path to save figure.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    learning_res = validation_results['learning_results']
    forgetting_res = validation_results['forgetting_results']
    spacing_res = validation_results['spacing_results']
    capacity_res = validation_results['capacity_results']

    # Overall score (large, top center)
    ax_score = fig.add_subplot(gs[0, 1])
    score = validation_results['overall_score']
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    color_idx = int(score * (len(colors) - 1))
    color = colors[color_idx]

    ax_score.bar([0], [score], color=color, alpha=0.7, width=0.5)
    ax_score.set_ylim(0, 1)
    ax_score.set_xlim(-0.5, 0.5)
    ax_score.set_xticks([])
    ax_score.set_ylabel('Validation Score')
    ax_score.set_title(f'Overall Validation: {score:.3f}', fontsize=14, fontweight='bold')
    ax_score.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Strong')
    ax_score.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    ax_score.legend()
    ax_score.grid(True, alpha=0.3, axis='y')

    # Learning curve correlation
    ax1 = fig.add_subplot(gs[0, 0])
    correlation = learning_res.get('rl_human_correlation', 0)
    ax1.bar(['RL-Human'], [correlation], color='steelblue', alpha=0.7)
    ax1.set_ylim(-1, 1)
    ax1.set_ylabel('Correlation')
    ax1.set_title('Learning Curve Correlation')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # Power law exponents
    ax2 = fig.add_subplot(gs[0, 2])
    human_exp = learning_res.get('human_power_law_b', 0)
    rl_exp = learning_res.get('rl_power_law_b', 0)
    ax2.bar(['Human', 'RL'], [human_exp, rl_exp], color=['coral', 'steelblue'], alpha=0.7)
    ax2.set_ylabel('Power Law Exponent')
    ax2.set_title('Learning Rate Comparison')
    ax2.grid(True, alpha=0.3)

    # Forgetting rates
    if 'error' not in forgetting_res:
        ax3 = fig.add_subplot(gs[1, 0])
        human_decay = forgetting_res.get('human_decay_rate', 0)
        rl_decay = forgetting_res.get('rl_decay_rate', 0)
        ax3.bar(['Human', 'RL'], [human_decay, rl_decay], color=['coral', 'steelblue'], alpha=0.7)
        ax3.set_ylabel('Decay Rate')
        ax3.set_title('Forgetting Rate Comparison')
        ax3.grid(True, alpha=0.3)

    # Spacing effect
    ax4 = fig.add_subplot(gs[1, 1])
    human_spacing = spacing_res.get('human_spacing_benefit', 0)
    rl_spacing = spacing_res.get('rl_spacing_benefit', 0)
    ax4.bar(['Human', 'RL'], [human_spacing, rl_spacing], color=['coral', 'steelblue'], alpha=0.7)
    ax4.set_ylabel('Benefit (%)')
    ax4.set_title('Spacing Effect')
    ax4.grid(True, alpha=0.3)

    # Memory capacity
    ax5 = fig.add_subplot(gs[1, 2])
    human_cap = capacity_res.get('human_capacity', 0)
    rl_cap = capacity_res.get('rl_capacity', 0)
    ax5.bar(['Human', 'RL'], [human_cap, rl_cap], color=['coral', 'steelblue'], alpha=0.7)
    ax5.set_ylabel('Items')
    ax5.set_title('Memory Capacity')
    ax5.axhline(y=7, color='g', linestyle='--', alpha=0.5, label='Miller\'s 7±2')
    ax5.axhline(y=5, color='g', linestyle='--', alpha=0.5)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Summary metrics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')

    table_data = [
        ['Metric', 'Human', 'RL Agent', 'Similarity'],
        ['Learning Correlation', '-', '-', f"{correlation:.3f}"],
        ['Power Law Exponent', f"{human_exp:.3f}", f"{rl_exp:.3f}",
         f"Diff: {abs(human_exp - rl_exp):.3f}"],
        ['Spacing Benefit (%)', f"{human_spacing:.1f}", f"{rl_spacing:.1f}",
         f"{spacing_res.get('spacing_similarity', 0):.3f}"],
        ['Memory Capacity', f"{human_cap:.1f}", f"{rl_cap:.1f}",
         f"Diff: {abs(human_cap - rl_cap):.2f}"]
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Header row styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle('Comprehensive Validation Analysis', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_data_distribution(
    data: pd.DataFrame,
    title: str = "Cognitive Task Data Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of data features.

    Args:
        data: DataFrame with cognitive data.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Performance distribution
    axes[0, 0].hist(data['performance'], bins=50, alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Recall Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Performance Distribution')

    # Cognitive load distribution
    axes[0, 1].hist(data['cognitive_load'], bins=50, alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('Cognitive Load')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Cognitive Load Distribution')

    # Reaction time distribution
    axes[0, 2].hist(data['reaction_time'], bins=50, alpha=0.7, color='green')
    axes[0, 2].set_xlabel('Reaction Time (ms)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reaction Time Distribution')

    # Sequence length vs performance
    axes[1, 0].scatter(data['sequence_length'], data['performance'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Performance')
    axes[1, 0].set_title('Length vs Performance')

    # Retention interval vs performance
    if 'retention_interval' in data.columns:
        axes[1, 1].scatter(data['retention_interval'], data['performance'], alpha=0.3, s=10)
        axes[1, 1].set_xlabel('Retention Interval (s)')
        axes[1, 1].set_ylabel('Performance')
        axes[1, 1].set_title('Retention vs Performance')
        axes[1, 1].set_xscale('log')

    # Repetitions vs performance
    if 'num_repetitions' in data.columns:
        axes[1, 2].scatter(data['num_repetitions'], data['performance'], alpha=0.3, s=10)
        axes[1, 2].set_xlabel('Number of Repetitions')
        axes[1, 2].set_ylabel('Performance')
        axes[1, 2].set_title('Practice vs Performance')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
