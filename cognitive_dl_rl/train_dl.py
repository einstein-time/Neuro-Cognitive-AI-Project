"""
Main script for training Deep Learning cognitive prediction model.

This script:
1. Generates cognitive task data
2. Creates and trains Cognitive Load Transformer
3. Evaluates and saves the model
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from config import get_default_config, save_config
from data.cognitive_data_generator import generate_cognitive_task_data
from data.preprocessing import split_data, create_dataloaders
from models.cognitive_transformer import create_cognitive_transformer
from training.dl_trainer import train_cognitive_model
from visualization.plots import plot_training_progress, plot_data_distribution


def main():
    """Main training function for DL model."""
    parser = argparse.ArgumentParser(description='Train Cognitive Load Transformer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get configuration
    config = get_default_config()
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.seed = args.seed

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, f"{args.results_dir}/config.json")

    print("=" * 70)
    print("PHASE 1: DEEP LEARNING - COGNITIVE PREDICTION MODEL")
    print("=" * 70)

    # Step 1: Generate cognitive task data
    print("\n1. Generating cognitive task data...")
    print("-" * 70)

    data = generate_cognitive_task_data(
        task_type='word_memory',
        n_participants=config.data.n_participants,
        seed=args.seed
    )

    print(f"Generated {len(data)} trials from {data['participant_id'].nunique()} participants")
    print(f"Data shape: {data.shape}")
    print(f"Features: {list(data.columns)}")

    # Visualize data distribution
    plot_data_distribution(data, save_path=f"{args.results_dir}/data_distribution.png")

    # Step 2: Split data
    print("\n2. Splitting data...")
    print("-" * 70)

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        seed=args.seed
    )

    print(f"Train: {len(train_data)} trials ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Val: {len(val_data)} trials ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test: {len(test_data)} trials ({len(test_data)/len(data)*100:.1f}%)")

    # Step 3: Create dataloaders
    print("\n3. Creating dataloaders...")
    print("-" * 70)

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_data,
        val_data,
        test_data,
        batch_size=config.training.batch_size,
        sequence_length=config.data.max_sequence_length,
        num_workers=config.training.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Step 4: Create model
    print("\n4. Creating Cognitive Load Transformer...")
    print("-" * 70)

    device = torch.device(config.training.device)
    print(f"Device: {device}")

    model_config = {
        'input_dim': 9,  # Number of input features
        'd_model': config.transformer.d_model,
        'nhead': config.transformer.nhead,
        'num_encoder_layers': config.transformer.num_encoder_layers,
        'dim_feedforward': config.transformer.dim_feedforward,
        'dropout': config.transformer.dropout,
        'max_sequence_length': config.transformer.max_sequence_length,
        'forgetting_rate': config.transformer.forgetting_rate
    }

    model = create_cognitive_transformer(model_config, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Step 5: Train model
    print("\n5. Training model...")
    print("-" * 70)

    training_config = {
        'num_epochs': config.training.num_epochs,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'batch_size': config.training.batch_size,
        'accuracy_weight': config.training.accuracy_weight,
        'reaction_time_weight': config.training.reaction_time_weight,
        'confidence_weight': config.training.confidence_weight,
        'gradient_clip_norm': config.training.gradient_clip_norm,
        'scheduler_type': config.training.scheduler_type,
        'patience': config.training.patience,
        'min_delta': config.training.min_delta,
        'save_best_model': config.training.save_best_model
    }

    trained_model, history = train_cognitive_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # Step 6: Plot training progress
    print("\n6. Visualizing training progress...")
    print("-" * 70)

    plot_training_progress(history, save_path=f"{args.results_dir}/training_progress.png")

    # Step 7: Evaluate on test set
    print("\n7. Evaluating on test set...")
    print("-" * 70)

    trained_model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            task_sequence = batch['task_sequence'].to(device)
            target_accuracy = batch['target_accuracy'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            pred_accuracy, _, _ = trained_model(task_sequence, attention_mask)

            valid_mask = ~attention_mask
            test_predictions.extend(pred_accuracy[valid_mask].cpu().numpy().flatten())
            test_targets.extend(target_accuracy[valid_mask].cpu().numpy().flatten())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    # Compute metrics
    test_correlation = np.corrcoef(test_predictions, test_targets)[0, 1]
    test_mae = np.mean(np.abs(test_predictions - test_targets))
    test_rmse = np.sqrt(np.mean((test_predictions - test_targets) ** 2))

    print(f"Test Correlation: {test_correlation:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Save results
    results = {
        'test_correlation': float(test_correlation),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'best_val_correlation': float(history['val_correlation'][-1])
    }

    import json
    with open(f"{args.results_dir}/dl_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE: Deep Learning Model Trained")
    print("=" * 70)
    print(f"Best validation correlation: {results['best_val_correlation']:.4f}")
    print(f"Test correlation: {results['test_correlation']:.4f}")
    print(f"Model saved to: {args.checkpoint_dir}/best_model.pth")
    print(f"Results saved to: {args.results_dir}/dl_results.json")


if __name__ == '__main__':
    main()
