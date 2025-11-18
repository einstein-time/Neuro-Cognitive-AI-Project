"""
Reinforcement Learning environment with human-like cognitive constraints.

This environment tests learning under constraints that mirror human cognition:
- Limited working memory capacity (Miller's 7±2)
- Attention bottleneck (can only process limited items per step)
- Exponential forgetting over time (Ebbinghaus curve)
- Cognitive load effects on performance

The hypothesis: RL agents trained in this environment will naturally develop
learning patterns that match human behavior, validating cognitive theory.
"""

import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class MemoryTrace:
    """Represents a memory trace with decay."""
    symbol: int
    strength: float
    time_since_study: int


class CognitiveTaskEnvironment(gym.Env):
    """
    RL environment for sequential learning with cognitive constraints.

    Task: Agent must learn to recall sequences of symbols (e.g., letters).

    Cognitive constraints:
    1. Limited working memory (capacity of 4-7 items)
    2. Attention bottleneck (can only attend to 1 item per timestep)
    3. Exponential forgetting (memory strength decays over time)
    4. Interference (new memories can displace old ones)

    The agent must learn optimal strategies for:
    - Attention allocation (what to rehearse)
    - Memory management (what to keep vs. forget)
    - Spacing practice (distributed vs. massed)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        sequence_length: int = 20,
        memory_capacity: int = 5,
        forgetting_rate: float = 0.01,
        attention_bottleneck: int = 1,
        n_symbols: int = 26,
        correct_reward: float = 1.0,
        incorrect_penalty: float = -0.5,
        time_penalty: float = -0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize cognitive task environment.

        Args:
            sequence_length: Number of symbols in sequence to learn.
            memory_capacity: Maximum number of items in working memory.
            forgetting_rate: Exponential decay rate (λ in R(t) = exp(-λt)).
            attention_bottleneck: Number of items that can be attended per step.
            n_symbols: Number of unique symbols (e.g., 26 for A-Z).
            correct_reward: Reward for correct recall.
            incorrect_penalty: Penalty for incorrect recall.
            time_penalty: Small penalty per timestep to encourage efficiency.
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.memory_capacity = memory_capacity
        self.forgetting_rate = forgetting_rate
        self.attention_bottleneck = attention_bottleneck
        self.n_symbols = n_symbols
        self.correct_reward = correct_reward
        self.incorrect_penalty = incorrect_penalty
        self.time_penalty = time_penalty

        # Set random seed
        self.rng = np.random.RandomState(seed)

        # Define observation space
        # Observation includes:
        # - Current symbol to encode/recall (one-hot)
        # - Memory state (strength of each memory slot)
        # - Phase indicator (study=0, test=1)
        # - Cognitive load (current memory usage / capacity)
        self.observation_space = spaces.Dict({
            'symbol': spaces.Box(0, 1, shape=(n_symbols,), dtype=np.float32),
            'memory_state': spaces.Box(0, 1, shape=(memory_capacity,), dtype=np.float32),
            'phase': spaces.Discrete(2),  # 0=study, 1=test
            'load': spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        })

        # Define action space
        # Actions: which memory slot to attend to (or skip)
        # Action i (i < capacity): Attend to slot i
        # Action capacity: Skip (don't attend)
        self.action_space = spaces.Discrete(memory_capacity + 1)

        # Internal state
        self.current_sequence = None
        self.memory_store: Dict[int, MemoryTrace] = {}
        self.current_step = 0
        self.study_phase = True
        self.test_step = 0

        # Performance tracking
        self.episode_correct = 0
        self.episode_total = 0

    def reset(self) -> Dict:
        """
        Reset environment for new episode.

        Returns:
            Initial observation.
        """
        # Generate random sequence to learn
        self.current_sequence = self.rng.randint(
            0, self.n_symbols, size=self.sequence_length
        )

        # Reset memory
        self.memory_store = {}
        self.current_step = 0
        self.study_phase = True
        self.test_step = 0

        # Reset performance tracking
        self.episode_correct = 0
        self.episode_total = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Action to take (which memory slot to attend to).

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Apply forgetting to all memories
        self._apply_forgetting()

        if self.study_phase:
            reward = self._study_step(action)
            done = False

            # Check if study phase is complete
            if self.current_step >= self.sequence_length:
                self.study_phase = False
                self.test_step = 0

        else:
            reward, done = self._test_step(action)

        self.current_step += 1

        # Add time penalty
        reward += self.time_penalty

        observation = self._get_observation()

        info = {
            'episode_correct': self.episode_correct,
            'episode_total': self.episode_total,
            'accuracy': self.episode_correct / max(1, self.episode_total),
            'memory_usage': len(self.memory_store) / self.memory_capacity
        }

        return observation, reward, done, info

    def _study_step(self, action: int) -> float:
        """
        Execute study phase step.

        Agent sees current symbol and decides which memory slot to update.

        Args:
            action: Memory slot to attend to.

        Returns:
            Reward for this step (0 during study).
        """
        current_symbol = self.current_sequence[self.current_step]

        # If action < memory_capacity, agent is attending to a slot
        if action < self.memory_capacity:
            # Check if at capacity
            if len(self.memory_store) >= self.memory_capacity:
                # Must forget weakest memory
                weakest_symbol = min(
                    self.memory_store.keys(),
                    key=lambda s: self.memory_store[s].strength
                )
                del self.memory_store[weakest_symbol]

            # Update or create memory trace
            if current_symbol in self.memory_store:
                # Strengthen existing memory (spacing effect)
                old_strength = self.memory_store[current_symbol].strength
                new_strength = min(1.0, old_strength + 0.2)
                self.memory_store[current_symbol] = MemoryTrace(
                    symbol=current_symbol,
                    strength=new_strength,
                    time_since_study=0
                )
            else:
                # Create new memory trace
                self.memory_store[current_symbol] = MemoryTrace(
                    symbol=current_symbol,
                    strength=0.5,
                    time_since_study=0
                )

        # No reward during study phase
        return 0.0

    def _test_step(self, action: int) -> Tuple[float, bool]:
        """
        Execute test phase step.

        Agent attempts to recall symbols from sequence.

        Args:
            action: Memory slot to retrieve from.

        Returns:
            Tuple of (reward, done).
        """
        target_symbol = self.current_sequence[self.test_step]

        # Check if symbol is in memory
        if target_symbol in self.memory_store:
            memory_strength = self.memory_store[target_symbol].strength

            # Recall probability based on memory strength
            recall_success = self.rng.random() < memory_strength

            if recall_success:
                reward = self.correct_reward
                self.episode_correct += 1
            else:
                reward = self.incorrect_penalty

        else:
            # Symbol not in memory
            reward = self.incorrect_penalty

        self.episode_total += 1
        self.test_step += 1

        # Episode ends after testing all symbols
        done = self.test_step >= self.sequence_length

        return reward, done

    def _apply_forgetting(self):
        """
        Apply exponential forgetting to all memory traces.

        Implements Ebbinghaus forgetting curve: R(t) = exp(-λt)
        """
        symbols_to_remove = []

        for symbol, trace in self.memory_store.items():
            # Increment time
            trace.time_since_study += 1

            # Apply exponential decay
            decay_factor = np.exp(-self.forgetting_rate * trace.time_since_study)
            trace.strength *= decay_factor

            # Remove if too weak
            if trace.strength < 0.05:
                symbols_to_remove.append(symbol)

        # Remove forgotten memories
        for symbol in symbols_to_remove:
            del self.memory_store[symbol]

    def _get_observation(self) -> Dict:
        """
        Get current observation.

        Returns:
            Observation dictionary.
        """
        # Current symbol (one-hot encoded)
        if self.study_phase:
            if self.current_step < len(self.current_sequence):
                current_symbol = self.current_sequence[self.current_step]
            else:
                current_symbol = 0
        else:
            if self.test_step < len(self.current_sequence):
                current_symbol = self.current_sequence[self.test_step]
            else:
                current_symbol = 0

        symbol_one_hot = np.zeros(self.n_symbols, dtype=np.float32)
        symbol_one_hot[current_symbol] = 1.0

        # Memory state (strength of each slot)
        memory_state = np.zeros(self.memory_capacity, dtype=np.float32)

        for idx, symbol in enumerate(list(self.memory_store.keys())[:self.memory_capacity]):
            memory_state[idx] = self.memory_store[symbol].strength

        # Phase indicator
        phase = 0 if self.study_phase else 1

        # Cognitive load
        load = np.array([len(self.memory_store) / self.memory_capacity], dtype=np.float32)

        return {
            'symbol': symbol_one_hot,
            'memory_state': memory_state,
            'phase': phase,
            'load': load
        }

    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Phase: {'Study' if self.study_phase else 'Test'}")
            print(f"Memory usage: {len(self.memory_store)}/{self.memory_capacity}")
            print("Memory contents:")
            for symbol, trace in self.memory_store.items():
                print(f"  Symbol {symbol}: strength={trace.strength:.2f}, "
                      f"time={trace.time_since_study}")
            print(f"Accuracy: {self.episode_correct}/{self.episode_total}")


class CurriculumCognitiveEnvironment(CognitiveTaskEnvironment):
    """
    Cognitive environment with curriculum learning.

    Gradually increases task difficulty:
    - Stage 1: Short sequences, high capacity
    - Stage 2: Medium sequences, medium capacity
    - Stage 3: Long sequences, realistic capacity
    - Stage 4+: Variable difficulty
    """

    def __init__(self, **kwargs):
        """Initialize with curriculum stages."""
        super().__init__(**kwargs)

        self.current_stage = 0
        self.episodes_in_stage = 0
        self.episodes_per_stage = 100

        # Curriculum stages (sequence_length, memory_capacity)
        self.stages = [
            (10, 7),   # Easy: short sequence, high capacity
            (15, 6),   # Medium: longer sequence, medium capacity
            (20, 5),   # Hard: long sequence, realistic capacity
            (25, 5),   # Very hard: very long sequence
            (30, 5),   # Expert: maximum difficulty
        ]

    def reset(self) -> Dict:
        """Reset with curriculum-based difficulty."""
        # Update difficulty based on curriculum stage
        if self.current_stage < len(self.stages):
            seq_len, capacity = self.stages[self.current_stage]
            self.sequence_length = seq_len
            self.memory_capacity = capacity

        # Check if ready for next stage
        self.episodes_in_stage += 1
        if self.episodes_in_stage >= self.episodes_per_stage:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.episodes_in_stage = 0
                print(f"\nProgressing to curriculum stage {self.current_stage + 1}")
                print(f"  Sequence length: {self.sequence_length}")
                print(f"  Memory capacity: {self.memory_capacity}")

        return super().reset()
