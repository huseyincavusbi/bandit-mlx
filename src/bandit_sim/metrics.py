"""Metrics utilities for analyzing bandit simulation results."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .simulation import SimulationResults


@dataclass
class AggregatedMetrics:
    """Summary statistics aggregated across multiple simulation runs."""

    mean_cumulative_reward: mx.array
    std_cumulative_reward: mx.array
    mean_cumulative_regret: mx.array
    std_cumulative_regret: mx.array
    arm_selection_frequencies: mx.array


def compute_metrics(results: SimulationResults) -> AggregatedMetrics:
    """Compute summary statistics from simulation results.

    Args:
        results: Raw simulation outputs containing per-run, per-step data.

    Returns:
        AggregatedMetrics with mean/std curves and arm selection frequencies.
    """
    # Compute mean and standard deviation across runs (axis=0)
    mean_cumulative_reward = mx.mean(results.cumulative_rewards, axis=0)
    std_cumulative_reward = mx.std(results.cumulative_rewards, axis=0)
    
    mean_cumulative_regret = mx.mean(results.cumulative_regret, axis=0)
    std_cumulative_regret = mx.std(results.cumulative_regret, axis=0)

    # Compute arm selection frequencies
    num_arms = int(mx.max(results.arm_selections).item()) + 1
    num_runs, horizon = results.arm_selections.shape
    
    # One-hot encode arm selections and sum across all timesteps and runs
    arm_indices = mx.arange(num_arms, dtype=mx.int32)
    selections_flat = results.arm_selections.reshape(-1)
    one_hot = mx.equal(mx.expand_dims(selections_flat, axis=1), arm_indices)
    counts = mx.sum(one_hot.astype(mx.float32), axis=0)
    
    total_selections = float(num_runs * horizon)
    frequencies = counts / total_selections

    return AggregatedMetrics(
        mean_cumulative_reward=mean_cumulative_reward,
        std_cumulative_reward=std_cumulative_reward,
        mean_cumulative_regret=mean_cumulative_regret,
        std_cumulative_regret=std_cumulative_regret,
        arm_selection_frequencies=frequencies,
    )


def compare_algorithms(results_dict: dict[str, SimulationResults]) -> dict[str, AggregatedMetrics]:
    """Compute metrics for multiple algorithms for easy comparison.

    Args:
        results_dict: Mapping from algorithm names to their simulation results.

    Returns:
        Dictionary mapping algorithm names to their aggregated metrics.
    """
    return {name: compute_metrics(results) for name, results in results_dict.items()}
