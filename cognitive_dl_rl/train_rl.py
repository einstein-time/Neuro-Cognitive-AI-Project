"""
Main script for training RL agent with cognitive constraints.

This script:
1. Creates cognitive environment with human-like constraints
2. Trains RL agent using PPO
3. Analyzes emergent learning patterns
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json

from config import get_default_config
from rl.cognitive_environment import CurriculumCognitiveEnvironment
from rl.cognitive_agent import CognitiveAgent, PPOTrainer
from rl.training import train_rl_agent, evaluate_agent, extract_learning_curve, analyze_spacing_patterns
from visualization.plots import plot_training_progress


def main():
    """Main training function for RL agent."""
    parser = argparse.ArgumentParser(description='Train RL Agent with Cognitive Constraints')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--update_freq', type=int, default=2048, help='Steps between updates')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_rl', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get configuration
    config = get_default_config()

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: REINFORCEMENT LEARNING - MECHANISTIC VALIDATION")
    print("=" * 70)

    # Step 1: Create environment
    print("\n1. Creating cognitive environment...")
    print("-" * 70)

    if args.curriculum:
        env = CurriculumCognitiveEnvironment(
            sequence_length=config.rl_env.sequence_length,
            memory_capacity=config.rl_env.memory_capacity,
            forgetting_rate=config.rl_env.forgetting_rate,
            attention_bottleneck=config.rl_env.attention_bottleneck,
            n_symbols=config.rl_env.n_symbols,
            correct_reward=config.rl_env.correct_recall_reward,
            incorrect_penalty=config.rl_env.incorrect_recall_penalty,
            time_penalty=config.rl_env.time_penalty,
            seed=args.seed
        )
        print("Using curriculum learning environment")
    else:
        from rl.cognitive_environment import CognitiveTaskEnvironment
        env = CognitiveTaskEnvironment(
            sequence_length=config.rl_env.sequence_length,
            memory_capacity=config.rl_env.memory_capacity,
            forgetting_rate=config.rl_env.forgetting_rate,
            attention_bottleneck=config.rl_env.attention_bottleneck,
            n_symbols=config.rl_env.n_symbols,
            correct_reward=config.rl_env.correct_recall_reward,
            incorrect_penalty=config.rl_env.incorrect_recall_penalty,
            time_penalty=config.rl_env.time_penalty,
            seed=args.seed
        )

    print(f"Environment: {env.__class__.__name__}")
    print(f"Sequence length: {env.sequence_length}")
    print(f"Memory capacity: {env.memory_capacity} items")
    print(f"Forgetting rate: {env.forgetting_rate}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Step 2: Create agent
    print("\n2. Creating cognitive agent...")
    print("-" * 70)

    device = torch.device(config.rl_agent.device)
    print(f"Device: {device}")

    agent = CognitiveAgent(
        n_symbols=config.rl_env.n_symbols,
        memory_capacity=config.rl_env.memory_capacity,
        hidden_dim=config.rl_agent.hidden_dim,
        num_layers=config.rl_agent.num_layers
    )

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {num_params:,}")

    # Create PPO trainer
    trainer = PPOTrainer(
        agent=agent,
        learning_rate=config.rl_agent.learning_rate,
        gamma=config.rl_agent.gamma,
        gae_lambda=config.rl_agent.gae_lambda,
        clip_epsilon=config.rl_agent.clip_epsilon,
        value_loss_coef=config.rl_agent.value_loss_coef,
        entropy_coef=config.rl_agent.entropy_coef,
        max_grad_norm=config.rl_agent.max_grad_norm,
        device=device
    )

    print("PPO trainer created")

    # Step 3: Train agent
    print("\n3. Training agent...")
    print("-" * 70)

    history = train_rl_agent(
        env=env,
        agent=agent,
        trainer=trainer,
        num_episodes=args.episodes,
        update_frequency=args.update_freq,
        num_epochs_per_update=config.rl_agent.num_epochs_per_update,
        batch_size=config.rl_agent.batch_size,
        eval_frequency=100,
        save_frequency=500,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True
    )

    # Step 4: Visualize training
    print("\n4. Visualizing training progress...")
    print("-" * 70)

    plot_training_progress(history, save_path=f"{args.results_dir}/rl_training_progress.png")

    # Step 5: Evaluate trained agent
    print("\n5. Evaluating trained agent...")
    print("-" * 70)

    eval_metrics = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=100,
        deterministic=True,
        verbose=False
    )

    print(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"Mean accuracy: {eval_metrics['mean_accuracy']:.3f} ± {eval_metrics['std_accuracy']:.3f}")
    print(f"Mean episode length: {eval_metrics['mean_length']:.1f}")

    # Step 6: Extract learning curve
    print("\n6. Analyzing emergent patterns...")
    print("-" * 70)

    rl_learning_curve = extract_learning_curve(history, window=100)
    print(f"Learning curve extracted: {len(rl_learning_curve)} points")

    # Analyze spacing patterns
    spacing_analysis = analyze_spacing_patterns(env, agent, num_episodes=50)
    print(f"Spacing score: {spacing_analysis['mean_spacing_score']:.3f} ± {spacing_analysis['std_spacing_score']:.3f}")

    # Step 7: Save results
    print("\n7. Saving results...")
    print("-" * 70)

    results = {
        'final_mean_reward': float(eval_metrics['mean_reward']),
        'final_mean_accuracy': float(eval_metrics['mean_accuracy']),
        'spacing_score': float(spacing_analysis['mean_spacing_score']),
        'learning_curve': rl_learning_curve.tolist(),
        'episode_accuracies': history['episode_accuracies']
    }

    with open(f"{args.results_dir}/rl_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save learning curve
    np.save(f"{args.results_dir}/rl_learning_curve.npy", rl_learning_curve)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE: RL Agent Trained")
    print("=" * 70)
    print(f"Final accuracy: {eval_metrics['mean_accuracy']:.3f}")
    print(f"Agent saved to: {args.checkpoint_dir}/agent_final.pth")
    print(f"Results saved to: {args.results_dir}/rl_results.json")
    print("\nKey observations:")
    print(f"- Agent achieves {eval_metrics['mean_accuracy']*100:.1f}% recall accuracy")
    print(f"- Spacing pattern score: {spacing_analysis['mean_spacing_score']:.2f}")
    print("- Ready for hypothesis validation (Phase 3)")


if __name__ == '__main__':
    main()
