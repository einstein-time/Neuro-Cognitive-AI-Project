"""
Cognitive task data generation based on established psychological models.

This module generates synthetic but psychologically realistic data for training
the Deep Learning prediction model. Data generation is based on:
- Ebbinghaus forgetting curve (exponential decay)
- Power Law of Learning (Newell & Rosenbloom, 1981)
- Cowan's working memory capacity model (2001)
- Cognitive Load Theory (Sweller, 1988)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CognitiveParameters:
    """Individual cognitive parameters for a simulated participant."""

    capacity: float  # Working memory capacity (items)
    forgetting_rate: float  # Exponential decay rate
    learning_rate: float  # Power law coefficient
    processing_speed: float  # Base reaction time (ms)
    attention_control: float  # Ability to manage cognitive load


class CognitiveDataGenerator:
    """
    Generate realistic cognitive task performance data.

    Based on established psychological models and incorporates:
    - Individual differences (normally distributed)
    - Learning curves (power law)
    - Forgetting curves (exponential decay)
    - Cognitive load effects
    - Spacing effects (distributed practice advantage)
    """

    def __init__(self, seed: int = 42):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

    def generate_participant_parameters(
        self,
        n_participants: int,
        mean_capacity: float = 5.0,
        std_capacity: float = 1.5
    ) -> List[CognitiveParameters]:
        """
        Generate individual cognitive parameters for simulated participants.

        Individual differences are normally distributed around population means,
        consistent with cognitive psychology research.

        Args:
            n_participants: Number of participants to simulate.
            mean_capacity: Mean working memory capacity (Cowan, 2001: ~4 items).
            std_capacity: Standard deviation of capacity.

        Returns:
            List of CognitiveParameters for each participant.
        """
        participants = []

        for _ in range(n_participants):
            capacity = max(2.0, self.rng.normal(mean_capacity, std_capacity))
            forgetting_rate = max(0.001, self.rng.normal(0.01, 0.005))
            learning_rate = max(0.1, self.rng.normal(0.3, 0.1))
            processing_speed = max(200, self.rng.normal(500, 100))
            attention_control = max(0.1, min(1.0, self.rng.normal(0.7, 0.15)))

            participants.append(CognitiveParameters(
                capacity=capacity,
                forgetting_rate=forgetting_rate,
                learning_rate=learning_rate,
                processing_speed=processing_speed,
                attention_control=attention_control
            ))

        return participants

    def compute_cognitive_load(
        self,
        sequence_length: int,
        task_complexity: float,
        capacity: float
    ) -> float:
        """
        Compute cognitive load based on task demands and capacity.

        Cognitive load increases with:
        - Sequence length (more items to remember)
        - Task complexity (intrinsic difficulty)
        - Lower capacity (individual differences)

        Args:
            sequence_length: Number of items in sequence.
            task_complexity: Intrinsic task difficulty (0-1).
            capacity: Individual working memory capacity.

        Returns:
            Cognitive load (0-1, where >1 indicates overload).
        """
        intrinsic_load = sequence_length / (capacity * 2)
        extraneous_load = task_complexity * 0.3
        total_load = intrinsic_load + extraneous_load

        return min(1.0, total_load)

    def compute_recall_accuracy(
        self,
        params: CognitiveParameters,
        cognitive_load: float,
        retention_interval: float,
        num_repetitions: int,
        spacing: float = 1.0
    ) -> float:
        """
        Compute recall accuracy based on cognitive models.

        Combines:
        - Power Law of Learning: Performance = A * N^B
        - Forgetting Curve: R(t) = exp(-λt)
        - Cognitive Load: Performance decreases with load
        - Spacing Effect: Distributed > massed practice

        Args:
            params: Individual cognitive parameters.
            cognitive_load: Current cognitive load (0-1).
            retention_interval: Time since study (seconds).
            num_repetitions: Number of practice trials.
            spacing: Spacing factor (>1 = distributed, <1 = massed).

        Returns:
            Recall accuracy (0-1).
        """
        # Power law of learning (practice effect)
        practice_benefit = params.learning_rate * (num_repetitions ** 0.3)

        # Ebbinghaus forgetting curve
        forgetting = np.exp(-params.forgetting_rate * retention_interval)

        # Cognitive load effect (overload impairs encoding)
        load_penalty = 1.0 - (cognitive_load * 0.5)

        # Spacing effect (distributed practice is better)
        spacing_benefit = 1.0 + (spacing - 1.0) * 0.2

        # Combine factors
        base_accuracy = 0.5  # Chance level
        accuracy = base_accuracy + practice_benefit * forgetting * load_penalty * spacing_benefit

        # Clip to valid range and add noise
        accuracy = np.clip(accuracy, 0.0, 1.0)
        accuracy += self.rng.normal(0, 0.05)  # Measurement noise

        return np.clip(accuracy, 0.0, 1.0)

    def compute_reaction_time(
        self,
        params: CognitiveParameters,
        cognitive_load: float,
        accuracy: float
    ) -> float:
        """
        Compute reaction time based on cognitive factors.

        Reaction time increases with:
        - Higher cognitive load (more processing required)
        - Lower accuracy (more difficult retrieval)
        - Individual processing speed

        Args:
            params: Individual cognitive parameters.
            cognitive_load: Current cognitive load (0-1).
            accuracy: Recall accuracy for this trial.

        Returns:
            Reaction time in milliseconds.
        """
        base_rt = params.processing_speed

        # Load increases RT
        load_factor = 1.0 + cognitive_load

        # Speed-accuracy tradeoff (lower accuracy = slower RT)
        accuracy_factor = 1.0 + (1.0 - accuracy) * 0.5

        # Compute RT with noise
        rt = base_rt * load_factor * accuracy_factor
        rt += self.rng.normal(0, 50)  # Response variability

        return max(100.0, rt)  # Minimum 100ms

    def compute_confidence(
        self,
        accuracy: float,
        cognitive_load: float,
        params: CognitiveParameters
    ) -> float:
        """
        Compute metacognitive confidence rating.

        Confidence correlates with accuracy but is affected by:
        - Metacognitive ability (attention_control)
        - Cognitive load (overconfidence under load)

        Args:
            accuracy: Actual recall accuracy.
            cognitive_load: Current cognitive load.
            params: Individual cognitive parameters.

        Returns:
            Confidence rating (0-1).
        """
        # Base confidence from actual accuracy
        base_confidence = accuracy

        # Metacognitive calibration
        calibration = params.attention_control
        confidence = base_confidence * calibration + (1 - calibration) * 0.5

        # Overconfidence under high load
        if cognitive_load > 0.7:
            confidence += 0.1

        # Add noise
        confidence += self.rng.normal(0, 0.08)

        return np.clip(confidence, 0.0, 1.0)

    def generate_word_memory_task(
        self,
        n_participants: int = 10000,
        trials_per_participant: int = 50,
        mean_capacity: float = 5.0,
        std_capacity: float = 1.5
    ) -> pd.DataFrame:
        """
        Generate word list memory task data (Ebbinghaus-style).

        Participants study lists of words with varying:
        - List length
        - Study time
        - Retention interval
        - Number of repetitions
        - Spacing of practice

        Args:
            n_participants: Number of simulated participants.
            trials_per_participant: Number of trials per participant.
            mean_capacity: Mean working memory capacity.
            std_capacity: Standard deviation of capacity.

        Returns:
            DataFrame with columns:
            - participant_id: Unique identifier
            - trial_number: Sequential trial index
            - sequence_length: Number of words in list
            - study_time: Seconds spent studying
            - retention_interval: Delay before test (seconds)
            - num_repetitions: Number of practice trials
            - spacing: Spacing factor (1.0 = even, >1 = distributed)
            - task_complexity: Normalized difficulty (0-1)
            - cognitive_load: Estimated cognitive load
            - performance: Recall accuracy (0-1)
            - reaction_time: Response latency (ms)
            - confidence: Metacognitive judgment (0-1)
        """
        # Generate participant parameters
        participants = self.generate_participant_parameters(
            n_participants, mean_capacity, std_capacity
        )

        data = []

        for pid, params in enumerate(participants):
            for trial in range(trials_per_participant):
                # Sample task parameters
                sequence_length = self.rng.randint(10, 31)
                study_time = self.rng.uniform(5, 30)
                retention_interval = self.rng.choice([0, 3600, 86400, 604800])  # 0h, 1h, 1d, 1w
                num_repetitions = self.rng.randint(1, 6)
                spacing = self.rng.choice([0.5, 1.0, 2.0])  # Massed, even, distributed
                task_complexity = self.rng.uniform(0.1, 0.9)

                # Compute cognitive load
                cognitive_load = self.compute_cognitive_load(
                    sequence_length, task_complexity, params.capacity
                )

                # Compute performance metrics
                performance = self.compute_recall_accuracy(
                    params, cognitive_load, retention_interval,
                    num_repetitions, spacing
                )

                reaction_time = self.compute_reaction_time(
                    params, cognitive_load, performance
                )

                confidence = self.compute_confidence(
                    performance, cognitive_load, params
                )

                # Store trial data
                data.append({
                    'participant_id': pid,
                    'trial_number': trial,
                    'sequence_length': sequence_length,
                    'study_time': study_time,
                    'retention_interval': retention_interval,
                    'num_repetitions': num_repetitions,
                    'spacing': spacing,
                    'task_complexity': task_complexity,
                    'cognitive_load': cognitive_load,
                    'performance': performance,
                    'reaction_time': reaction_time,
                    'confidence': confidence,
                    'capacity': params.capacity,
                    'forgetting_rate': params.forgetting_rate
                })

        return pd.DataFrame(data)

    def generate_n_back_task(
        self,
        n_participants: int = 5000,
        trials_per_participant: int = 100
    ) -> pd.DataFrame:
        """
        Generate N-back working memory task data.

        Participants monitor a sequence and indicate when current item
        matches item N positions back.

        Args:
            n_participants: Number of simulated participants.
            trials_per_participant: Number of trials per participant.

        Returns:
            DataFrame with N-back task performance data.
        """
        participants = self.generate_participant_parameters(n_participants)

        data = []

        for pid, params in enumerate(participants):
            for trial in range(trials_per_participant):
                n_back_level = self.rng.choice([1, 2, 3])
                sequence_length = 20 + n_back_level * 10
                stimulus_speed = self.rng.uniform(1.0, 3.0)  # seconds per item

                # Cognitive load increases with N-back level
                task_complexity = n_back_level / 3.0
                cognitive_load = self.compute_cognitive_load(
                    n_back_level, task_complexity, params.capacity
                )

                # Performance decreases with higher N
                base_accuracy = max(0.3, 0.9 - n_back_level * 0.2)
                performance = base_accuracy * (1.0 - cognitive_load * 0.3)
                performance += self.rng.normal(0, 0.1)
                performance = np.clip(performance, 0.0, 1.0)

                reaction_time = self.compute_reaction_time(
                    params, cognitive_load, performance
                )

                confidence = self.compute_confidence(
                    performance, cognitive_load, params
                )

                data.append({
                    'participant_id': pid,
                    'trial_number': trial,
                    'n_back_level': n_back_level,
                    'sequence_length': sequence_length,
                    'stimulus_speed': stimulus_speed,
                    'cognitive_load': cognitive_load,
                    'performance': performance,
                    'reaction_time': reaction_time,
                    'confidence': confidence
                })

        return pd.DataFrame(data)


def generate_cognitive_task_data(
    task_type: str = 'word_memory',
    n_participants: int = 10000,
    seed: int = 42,
    **kwargs
) -> pd.DataFrame:
    """
    Generate cognitive task performance data.

    Convenience function for generating different task types.

    Args:
        task_type: Type of cognitive task ('word_memory' or 'n_back').
        n_participants: Number of simulated participants.
        seed: Random seed for reproducibility.
        **kwargs: Additional task-specific parameters.

    Returns:
        DataFrame with cognitive task performance data.
    """
    generator = CognitiveDataGenerator(seed=seed)

    if task_type == 'word_memory':
        return generator.generate_word_memory_task(n_participants, **kwargs)
    elif task_type == 'n_back':
        return generator.generate_n_back_task(n_participants, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
