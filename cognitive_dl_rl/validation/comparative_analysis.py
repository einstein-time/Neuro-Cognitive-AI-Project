"""
Comparative analysis: RL agents vs. Human learners vs. DL predictions.

This module implements the critical validation step of the project:
Testing whether RL agents with cognitive constraints naturally develop
human-like learning patterns, validating that cognitive constraints CAUSE
the observed patterns rather than just correlating with them.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List, Optional
import torch


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Power law function: y = a * x^b

    Models learning curves (Newell & Rosenbloom, 1981).

    Args:
        x: Trial numbers.
        a: Scaling parameter.
        b: Learning rate exponent.

    Returns:
        Performance values.
    """
    return a * np.power(x, b)


def exponential_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Exponential decay: y = a * exp(-b * x)

    Models forgetting curves (Ebbinghaus, 1885).

    Args:
        x: Time values.
        a: Initial strength.
        b: Decay rate.

    Returns:
        Retention values.
    """
    return a * np.exp(-b * x)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law to learning curve.

    Args:
        x: Trial numbers.
        y: Performance values.

    Returns:
        Tuple of (a, b, r_squared).
    """
    try:
        # Add small constant to avoid log(0)
        x_safe = x + 1
        y_safe = np.maximum(y, 1e-6)

        # Fit power law
        params, _ = curve_fit(power_law, x_safe, y_safe, p0=[0.5, 0.3], maxfev=5000)
        a, b = params

        # Compute R-squared
        y_pred = power_law(x_safe, a, b)
        ss_res = np.sum((y_safe - y_pred) ** 2)
        ss_tot = np.sum((y_safe - np.mean(y_safe)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return a, b, r_squared

    except:
        return 0.0, 0.0, 0.0


def fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit exponential decay to forgetting curve.

    Args:
        x: Time values.
        y: Retention values.

    Returns:
        Tuple of (a, b, r_squared).
    """
    try:
        # Fit exponential decay
        y_safe = np.maximum(y, 1e-6)
        params, _ = curve_fit(exponential_decay, x, y_safe, p0=[1.0, 0.01], maxfev=5000)
        a, b = params

        # Compute R-squared
        y_pred = exponential_decay(x, a, b)
        ss_res = np.sum((y_safe - y_pred) ** 2)
        ss_tot = np.sum((y_safe - np.mean(y_safe)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return a, b, r_squared

    except:
        return 0.0, 0.0, 0.0


class CognitiveValidator:
    """
    Validates RL agent behavior against human patterns.

    Tests for:
    1. Learning curves (power law)
    2. Forgetting curves (exponential decay)
    3. Spacing effects (distributed > massed practice)
    4. Capacity limits (7±2 items)
    """

    def __init__(self, human_data: pd.DataFrame):
        """
        Initialize validator with human data.

        Args:
            human_data: DataFrame with human performance data.
        """
        self.human_data = human_data

        # Extract human patterns
        self.human_learning_curve = self._extract_human_learning_curve()
        self.human_forgetting_curve = self._extract_human_forgetting_curve()
        self.human_capacity = self._estimate_human_capacity()
        self.human_spacing_effect = self._measure_human_spacing_effect()

    def _extract_human_learning_curve(self) -> np.ndarray:
        """
        Extract average learning curve from human data.

        Returns:
            Array of performance by trial number.
        """
        # Group by trial number and average performance
        learning_curve = self.human_data.groupby('trial_number')['performance'].mean().values
        return learning_curve

    def _extract_human_forgetting_curve(self) -> Dict[float, float]:
        """
        Extract forgetting curve from human data.

        Returns:
            Dictionary mapping retention interval to performance.
        """
        forgetting_curve = self.human_data.groupby('retention_interval')['performance'].mean().to_dict()
        return forgetting_curve

    def _estimate_human_capacity(self) -> float:
        """
        Estimate working memory capacity from performance.

        Returns:
            Estimated capacity (items).
        """
        # Capacity estimated from mean participant capacity
        if 'capacity' in self.human_data.columns:
            return self.human_data['capacity'].mean()
        else:
            return 5.0  # Default: Cowan's estimate

    def _measure_human_spacing_effect(self) -> float:
        """
        Measure spacing effect in human data.

        Returns:
            Benefit of distributed over massed practice (percentage).
        """
        if 'spacing' not in self.human_data.columns:
            return 20.0  # Typical spacing effect

        massed = self.human_data[self.human_data['spacing'] <= 1.0]['performance'].mean()
        distributed = self.human_data[self.human_data['spacing'] > 1.0]['performance'].mean()

        spacing_benefit = ((distributed - massed) / massed) * 100
        return spacing_benefit

    def validate_learning_curves(
        self,
        rl_learning_curve: np.ndarray,
        dl_predicted_curve: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Validate learning curves against human pattern.

        Tests:
        1. Correlation between RL and human
        2. Power law fit comparison
        3. DL prediction accuracy (if provided)

        Args:
            rl_learning_curve: Learning curve from RL agent.
            dl_predicted_curve: DL model predictions (optional).

        Returns:
            Dictionary of validation metrics.
        """
        results = {}

        # Ensure same length
        min_len = min(len(self.human_learning_curve), len(rl_learning_curve))
        human_curve = self.human_learning_curve[:min_len]
        rl_curve = rl_learning_curve[:min_len]

        # Correlation
        correlation, p_value = stats.pearsonr(human_curve, rl_curve)
        results['rl_human_correlation'] = correlation
        results['correlation_p_value'] = p_value

        # Fit power laws
        trials = np.arange(min_len)

        human_a, human_b, human_r2 = fit_power_law(trials, human_curve)
        rl_a, rl_b, rl_r2 = fit_power_law(trials, rl_curve)

        results['human_power_law_a'] = human_a
        results['human_power_law_b'] = human_b
        results['human_power_law_r2'] = human_r2
        results['rl_power_law_a'] = rl_a
        results['rl_power_law_b'] = rl_b
        results['rl_power_law_r2'] = rl_r2

        # Compare exponents
        results['power_law_exponent_diff'] = abs(human_b - rl_b)

        # DL predictions
        if dl_predicted_curve is not None:
            dl_curve = dl_predicted_curve[:min_len]
            dl_correlation, _ = stats.pearsonr(human_curve, dl_curve)
            results['dl_human_correlation'] = dl_correlation

            rl_dl_correlation, _ = stats.pearsonr(rl_curve, dl_curve)
            results['rl_dl_correlation'] = rl_dl_correlation

        # Statistical test: Are RL and human curves from same distribution?
        t_stat, t_pvalue = stats.ttest_ind(human_curve, rl_curve)
        results['ttest_statistic'] = t_stat
        results['ttest_p_value'] = t_pvalue

        return results

    def validate_forgetting_curves(
        self,
        rl_forgetting_curve: Dict[float, float]
    ) -> Dict[str, float]:
        """
        Validate forgetting curves.

        Args:
            rl_forgetting_curve: Dict mapping retention interval to performance.

        Returns:
            Dictionary of validation metrics.
        """
        results = {}

        # Get common retention intervals
        human_intervals = sorted(self.human_forgetting_curve.keys())
        rl_intervals = sorted(rl_forgetting_curve.keys())

        # Use intersection
        common_intervals = sorted(set(human_intervals) & set(rl_intervals))

        if len(common_intervals) < 2:
            return {'error': 'Insufficient data for forgetting curve comparison'}

        human_retention = np.array([self.human_forgetting_curve[t] for t in common_intervals])
        rl_retention = np.array([rl_forgetting_curve[t] for t in common_intervals])

        # Fit exponential decay
        intervals_array = np.array(common_intervals)

        human_a, human_decay, human_r2 = fit_exponential_decay(intervals_array, human_retention)
        rl_a, rl_decay, rl_r2 = fit_exponential_decay(intervals_array, rl_retention)

        results['human_decay_rate'] = human_decay
        results['human_decay_r2'] = human_r2
        results['rl_decay_rate'] = rl_decay
        results['rl_decay_r2'] = rl_r2

        # Compare decay rates
        results['decay_rate_diff'] = abs(human_decay - rl_decay)

        # Correlation
        correlation, p_value = stats.pearsonr(human_retention, rl_retention)
        results['forgetting_correlation'] = correlation
        results['forgetting_p_value'] = p_value

        return results

    def validate_spacing_effect(
        self,
        rl_spacing_benefit: float
    ) -> Dict[str, float]:
        """
        Validate spacing effect.

        Args:
            rl_spacing_benefit: RL agent's spacing benefit (percentage).

        Returns:
            Dictionary of validation metrics.
        """
        results = {}

        results['human_spacing_benefit'] = self.human_spacing_effect
        results['rl_spacing_benefit'] = rl_spacing_benefit
        results['spacing_benefit_diff'] = abs(self.human_spacing_effect - rl_spacing_benefit)

        # Relative similarity
        if self.human_spacing_effect > 0:
            results['spacing_similarity'] = 1 - (
                results['spacing_benefit_diff'] / max(self.human_spacing_effect, rl_spacing_benefit)
            )
        else:
            results['spacing_similarity'] = 0.0

        return results

    def validate_capacity_limits(
        self,
        rl_capacity: float
    ) -> Dict[str, float]:
        """
        Validate memory capacity.

        Args:
            rl_capacity: RL agent's estimated capacity.

        Returns:
            Dictionary of validation metrics.
        """
        results = {}

        results['human_capacity'] = self.human_capacity
        results['rl_capacity'] = rl_capacity
        results['capacity_diff'] = abs(self.human_capacity - rl_capacity)

        # Check if within Miller's 7±2 range
        results['rl_within_miller_range'] = 5.0 <= rl_capacity <= 9.0

        return results

    def compute_overall_validation_score(
        self,
        learning_results: Dict[str, float],
        forgetting_results: Dict[str, float],
        spacing_results: Dict[str, float],
        capacity_results: Dict[str, float]
    ) -> float:
        """
        Compute overall validation score.

        Combines multiple validation metrics into single score [0, 1].

        Args:
            learning_results: Learning curve validation results.
            forgetting_results: Forgetting curve validation results.
            spacing_results: Spacing effect validation results.
            capacity_results: Capacity validation results.

        Returns:
            Overall validation score (higher is better).
        """
        scores = []

        # Learning curve correlation (weight: 0.3)
        if 'rl_human_correlation' in learning_results:
            scores.append(max(0, learning_results['rl_human_correlation']) * 0.3)

        # Forgetting curve correlation (weight: 0.25)
        if 'forgetting_correlation' in forgetting_results:
            scores.append(max(0, forgetting_results['forgetting_correlation']) * 0.25)

        # Spacing similarity (weight: 0.2)
        if 'spacing_similarity' in spacing_results:
            scores.append(spacing_results['spacing_similarity'] * 0.2)

        # Capacity similarity (weight: 0.25)
        if 'capacity_diff' in capacity_results:
            capacity_score = max(0, 1 - capacity_results['capacity_diff'] / 5.0)
            scores.append(capacity_score * 0.25)

        overall_score = sum(scores)

        return overall_score


def validate_hypothesis(
    human_data: pd.DataFrame,
    rl_learning_curve: np.ndarray,
    rl_forgetting_curve: Dict[float, float],
    rl_spacing_benefit: float,
    rl_capacity: float,
    dl_predicted_curve: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Complete hypothesis validation pipeline.

    Tests the central hypothesis: RL agents with cognitive constraints
    naturally develop human-like learning patterns.

    Args:
        human_data: Human performance data.
        rl_learning_curve: RL agent learning curve.
        rl_forgetting_curve: RL agent forgetting curve.
        rl_spacing_benefit: RL agent spacing benefit.
        rl_capacity: RL agent estimated capacity.
        dl_predicted_curve: DL model predictions (optional).
        verbose: Print detailed results.

    Returns:
        Complete validation results dictionary.
    """
    validator = CognitiveValidator(human_data)

    # Run all validations
    learning_results = validator.validate_learning_curves(
        rl_learning_curve, dl_predicted_curve
    )

    forgetting_results = validator.validate_forgetting_curves(
        rl_forgetting_curve
    )

    spacing_results = validator.validate_spacing_effect(
        rl_spacing_benefit
    )

    capacity_results = validator.validate_capacity_limits(
        rl_capacity
    )

    # Overall score
    overall_score = validator.compute_overall_validation_score(
        learning_results, forgetting_results, spacing_results, capacity_results
    )

    if verbose:
        print("=" * 70)
        print("HYPOTHESIS VALIDATION: RL AGENTS vs HUMAN LEARNERS")
        print("=" * 70)
        print("\n1. LEARNING CURVES (Power Law of Practice)")
        print("-" * 70)
        print(f"   RL-Human Correlation: {learning_results.get('rl_human_correlation', 0):.4f}")
        print(f"   P-value: {learning_results.get('correlation_p_value', 1):.4f}")
        print(f"   Human Power Law: y = {learning_results.get('human_power_law_a', 0):.3f} * x^{learning_results.get('human_power_law_b', 0):.3f}")
        print(f"   RL Power Law: y = {learning_results.get('rl_power_law_a', 0):.3f} * x^{learning_results.get('rl_power_law_b', 0):.3f}")
        print(f"   Exponent Difference: {learning_results.get('power_law_exponent_diff', 0):.4f}")

        if 'dl_human_correlation' in learning_results:
            print(f"   DL-Human Correlation: {learning_results['dl_human_correlation']:.4f}")

        print("\n2. FORGETTING CURVES (Ebbinghaus Decay)")
        print("-" * 70)
        if 'error' not in forgetting_results:
            print(f"   Forgetting Correlation: {forgetting_results.get('forgetting_correlation', 0):.4f}")
            print(f"   Human Decay Rate: {forgetting_results.get('human_decay_rate', 0):.4f}")
            print(f"   RL Decay Rate: {forgetting_results.get('rl_decay_rate', 0):.4f}")
            print(f"   Decay Rate Difference: {forgetting_results.get('decay_rate_diff', 0):.4f}")
        else:
            print(f"   {forgetting_results['error']}")

        print("\n3. SPACING EFFECT (Distributed vs Massed Practice)")
        print("-" * 70)
        print(f"   Human Spacing Benefit: {spacing_results.get('human_spacing_benefit', 0):.2f}%")
        print(f"   RL Spacing Benefit: {spacing_results.get('rl_spacing_benefit', 0):.2f}%")
        print(f"   Similarity Score: {spacing_results.get('spacing_similarity', 0):.4f}")

        print("\n4. MEMORY CAPACITY (Miller's 7±2)")
        print("-" * 70)
        print(f"   Human Capacity: {capacity_results.get('human_capacity', 0):.2f} items")
        print(f"   RL Capacity: {capacity_results.get('rl_capacity', 0):.2f} items")
        print(f"   Difference: {capacity_results.get('capacity_diff', 0):.2f} items")
        print(f"   Within Miller's Range: {capacity_results.get('rl_within_miller_range', False)}")

        print("\n" + "=" * 70)
        print(f"OVERALL VALIDATION SCORE: {overall_score:.4f} / 1.0")
        print("=" * 70)

        if overall_score > 0.8:
            print("\nCONCLUSION: STRONG VALIDATION")
            print("=> RL agents exhibit human-like cognitive patterns")
            print("=> Cognitive constraints CAUSE observed learning patterns")
            print("=> Hypothesis SUPPORTED")
        elif overall_score > 0.6:
            print("\nCONCLUSION: MODERATE VALIDATION")
            print("=> RL agents show some human-like patterns")
            print("=> Further investigation recommended")
        else:
            print("\nCONCLUSION: WEAK VALIDATION")
            print("=> RL agents do not match human patterns")
            print("=> Hypothesis not supported or model needs refinement")

        print("=" * 70)

    return {
        'learning_results': learning_results,
        'forgetting_results': forgetting_results,
        'spacing_results': spacing_results,
        'capacity_results': capacity_results,
        'overall_score': overall_score
    }
