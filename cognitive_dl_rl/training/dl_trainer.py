"""
Training procedures for Cognitive Load Transformer model.

This module implements:
- Multi-task loss computation
- Training loop with validation
- Early stopping and checkpointing
- Learning rate scheduling
- Training metrics and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import numpy as np
from pathlib import Path
import json


class CognitiveModelTrainer:
    """
    Trainer for Cognitive Load Transformer model.

    Handles:
    - Multi-task learning (accuracy, RT, confidence)
    - Weighted loss combination
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize trainer.

        Args:
            model: CognitiveLoadTransformer model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration dictionary.
            device: Device for training.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss weights
        self.loss_weights = {
            'accuracy': config.get('accuracy_weight', 1.0),
            'reaction_time': config.get('reaction_time_weight', 0.5),
            'confidence': config.get('confidence_weight', 0.3)
        }

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler(config.get('scheduler_type', 'cosine'))

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_correlation = 0.0
        self.patience_counter = 0

        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_correlation': [],
            'learning_rate': []
        }

    def _create_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler."""
        num_epochs = self.config.get('num_epochs', 50)
        warmup_epochs = self.config.get('warmup_epochs', 5)

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def compute_multi_task_loss(
        self,
        pred_accuracy: torch.Tensor,
        pred_rt: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_accuracy: torch.Tensor,
        target_rt: torch.Tensor,
        target_confidence: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            pred_accuracy: Predicted accuracy [batch, seq_len, 1].
            pred_rt: Predicted reaction time [batch, seq_len, 1].
            pred_confidence: Predicted confidence [batch, seq_len, 1].
            target_accuracy: True accuracy [batch, seq_len, 1].
            target_rt: True reaction time [batch, seq_len, 1].
            target_confidence: True confidence [batch, seq_len, 1].
            mask: Attention mask [batch, seq_len], True = masked.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Invert mask (True = valid, False = masked)
        valid_mask = ~mask.unsqueeze(-1)  # [batch, seq_len, 1]

        # Compute losses only on valid positions
        loss_acc = F.mse_loss(
            pred_accuracy * valid_mask,
            target_accuracy * valid_mask,
            reduction='sum'
        ) / valid_mask.sum()

        loss_rt = F.mse_loss(
            pred_rt * valid_mask,
            target_rt * valid_mask,
            reduction='sum'
        ) / valid_mask.sum()

        loss_conf = F.mse_loss(
            pred_confidence * valid_mask,
            target_confidence * valid_mask,
            reduction='sum'
        ) / valid_mask.sum()

        # Weighted combination
        total_loss = (
            self.loss_weights['accuracy'] * loss_acc +
            self.loss_weights['reaction_time'] * loss_rt +
            self.loss_weights['confidence'] * loss_conf
        )

        loss_dict = {
            'accuracy': loss_acc.item(),
            'reaction_time': loss_rt.item(),
            'confidence': loss_conf.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()

        epoch_losses = {
            'accuracy': 0.0,
            'reaction_time': 0.0,
            'confidence': 0.0,
            'total': 0.0
        }

        num_batches = 0

        for batch in self.train_loader:
            # Move data to device
            task_sequence = batch['task_sequence'].to(self.device)
            target_accuracy = batch['target_accuracy'].to(self.device)
            target_rt = batch['target_rt'].to(self.device)
            target_confidence = batch['target_confidence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass
            pred_accuracy, pred_rt, pred_confidence = self.model(
                task_sequence,
                attention_mask=attention_mask
            )

            # Compute loss
            loss, loss_dict = self.compute_multi_task_loss(
                pred_accuracy, pred_rt, pred_confidence,
                target_accuracy, target_rt, target_confidence,
                attention_mask
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )

            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self) -> Tuple[Dict[str, float], float]:
        """
        Validate model on validation set.

        Returns:
            Tuple of (loss_dict, correlation).
        """
        self.model.eval()

        epoch_losses = {
            'accuracy': 0.0,
            'reaction_time': 0.0,
            'confidence': 0.0,
            'total': 0.0
        }

        all_predictions = []
        all_targets = []

        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                task_sequence = batch['task_sequence'].to(self.device)
                target_accuracy = batch['target_accuracy'].to(self.device)
                target_rt = batch['target_rt'].to(self.device)
                target_confidence = batch['target_confidence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                pred_accuracy, pred_rt, pred_confidence = self.model(
                    task_sequence,
                    attention_mask=attention_mask
                )

                # Compute loss
                loss, loss_dict = self.compute_multi_task_loss(
                    pred_accuracy, pred_rt, pred_confidence,
                    target_accuracy, target_rt, target_confidence,
                    attention_mask
                )

                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += loss_dict[key]

                # Collect predictions for correlation
                valid_mask = ~attention_mask
                pred_flat = pred_accuracy[valid_mask].cpu().numpy()
                target_flat = target_accuracy[valid_mask].cpu().numpy()

                all_predictions.extend(pred_flat.flatten())
                all_targets.extend(target_flat.flatten())

                num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Compute correlation
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]

        return epoch_losses, correlation

    def train(self) -> Dict[str, List[float]]:
        """
        Train model for configured number of epochs.

        Returns:
            Training history dictionary.
        """
        num_epochs = self.config.get('num_epochs', 50)
        patience = self.config.get('patience', 10)
        min_delta = self.config.get('min_delta', 0.001)

        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 32)}")
        print(f"Learning rate: {self.config.get('learning_rate', 1e-4)}")
        print("-" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_losses = self.train_epoch()

            # Validate
            val_losses, val_correlation = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_correlation'].append(val_correlation)
            self.history['learning_rate'].append(current_lr)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Acc: {train_losses['accuracy']:.4f}, "
                  f"RT: {train_losses['reaction_time']:.4f}, "
                  f"Conf: {train_losses['confidence']:.4f})")
            print(f"  Val Loss: {val_losses['total']:.4f} "
                  f"(Correlation: {val_correlation:.4f})")
            print(f"  LR: {current_lr:.6f}")

            # Check for improvement
            improved = False

            if val_correlation > self.best_val_correlation + min_delta:
                self.best_val_correlation = val_correlation
                self.best_val_loss = val_losses['total']
                improved = True
                self.patience_counter = 0

                # Save best model
                if self.config.get('save_best_model', True):
                    self.save_checkpoint('best_model.pth')
                    print("  Saved best model")

            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation correlation: {self.best_val_correlation:.4f}")
                break

            print("-" * 60)

        print("\nTraining completed!")
        print(f"Best validation correlation: {self.best_val_correlation:.4f}")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_correlation': self.best_val_correlation,
            'history': self.history,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_correlation = checkpoint['best_val_correlation']
        self.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best validation correlation: {self.best_val_correlation:.4f}")


def train_cognitive_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    checkpoint_dir: str = "checkpoints"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Convenience function to train cognitive model.

    Args:
        model: CognitiveLoadTransformer model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Device for training.
        checkpoint_dir: Directory for checkpoints.

    Returns:
        Tuple of (trained_model, training_history).
    """
    trainer = CognitiveModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    history = trainer.train()

    # Load best model
    trainer.load_checkpoint('best_model.pth')

    return trainer.model, history
