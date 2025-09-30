"""Visualization utilities for bandit simulation results."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from .metrics import AggregatedMetrics


def plot_cumulative_regret(
    metrics_dict: dict[str, AggregatedMetrics],
    *,
    title: str = "Cumulative Regret Comparison",
    xlabel: str = "Time Step",
    ylabel: str = "Cumulative Regret",
    show_std: bool = True,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot cumulative regret curves for multiple algorithms.

    Args:
        metrics_dict: Mapping from algorithm names to their metrics.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        show_std: Whether to show shaded standard deviation bands.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for algo_name, metrics in metrics_dict.items():
        mean_regret = np.array(metrics.mean_cumulative_regret)
        std_regret = np.array(metrics.std_cumulative_regret)
        timesteps = np.arange(len(mean_regret))

        ax.plot(timesteps, mean_regret, label=algo_name, linewidth=2)

        if show_std:
            ax.fill_between(
                timesteps,
                mean_regret - std_regret,
                mean_regret + std_regret,
                alpha=0.2,
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_cumulative_reward(
    metrics_dict: dict[str, AggregatedMetrics],
    *,
    title: str = "Cumulative Reward Comparison",
    xlabel: str = "Time Step",
    ylabel: str = "Cumulative Reward",
    show_std: bool = True,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot cumulative reward curves for multiple algorithms.

    Args:
        metrics_dict: Mapping from algorithm names to their metrics.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        show_std: Whether to show shaded standard deviation bands.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for algo_name, metrics in metrics_dict.items():
        mean_reward = np.array(metrics.mean_cumulative_reward)
        std_reward = np.array(metrics.std_cumulative_reward)
        timesteps = np.arange(len(mean_reward))

        ax.plot(timesteps, mean_reward, label=algo_name, linewidth=2)

        if show_std:
            ax.fill_between(
                timesteps,
                mean_reward - std_reward,
                mean_reward + std_reward,
                alpha=0.2,
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_arm_frequencies(
    metrics_dict: dict[str, AggregatedMetrics],
    *,
    num_arms: int,
    title: str = "Arm Selection Frequencies",
    xlabel: str = "Arm Index",
    ylabel: str = "Selection Frequency",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot arm selection frequency bars for multiple algorithms.

    Args:
        metrics_dict: Mapping from algorithm names to their metrics.
        num_arms: Number of arms in the bandit.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    num_algos = len(metrics_dict)
    bar_width = 0.8 / num_algos
    arm_indices = np.arange(num_arms)

    for i, (algo_name, metrics) in enumerate(metrics_dict.items()):
        frequencies = np.array(metrics.arm_selection_frequencies)
        offset = (i - num_algos / 2 + 0.5) * bar_width
        ax.bar(arm_indices + offset, frequencies, bar_width, label=algo_name, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(arm_indices)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


def create_comparison_dashboard(
    metrics_dict: dict[str, AggregatedMetrics],
    num_arms: int,
    *,
    suptitle: Optional[str] = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Create a multi-panel dashboard comparing algorithms.

    Args:
        metrics_dict: Mapping from algorithm names to their metrics.
        num_arms: Number of arms in the bandit.
        suptitle: Optional super-title for the entire figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object with subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Cumulative Regret
    ax = axes[0, 0]
    for algo_name, metrics in metrics_dict.items():
        mean_regret = np.array(metrics.mean_cumulative_regret)
        std_regret = np.array(metrics.std_cumulative_regret)
        timesteps = np.arange(len(mean_regret))
        ax.plot(timesteps, mean_regret, label=algo_name, linewidth=2)
        ax.fill_between(timesteps, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative Reward
    ax = axes[0, 1]
    for algo_name, metrics in metrics_dict.items():
        mean_reward = np.array(metrics.mean_cumulative_reward)
        std_reward = np.array(metrics.std_cumulative_reward)
        timesteps = np.arange(len(mean_reward))
        ax.plot(timesteps, mean_reward, label=algo_name, linewidth=2)
        ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Arm Selection Frequencies
    ax = axes[1, 0]
    num_algos = len(metrics_dict)
    bar_width = 0.8 / num_algos
    arm_indices = np.arange(num_arms)
    for i, (algo_name, metrics) in enumerate(metrics_dict.items()):
        frequencies = np.array(metrics.arm_selection_frequencies)
        offset = (i - num_algos / 2 + 0.5) * bar_width
        ax.bar(arm_indices + offset, frequencies, bar_width, label=algo_name, alpha=0.8)
    ax.set_xlabel("Arm Index")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Arm Selection Frequencies", fontweight="bold")
    ax.set_xticks(arm_indices)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Plot 4: Final Regret Comparison (bar chart)
    ax = axes[1, 1]
    algo_names = list(metrics_dict.keys())
    final_regrets = [np.array(m.mean_cumulative_regret)[-1] for m in metrics_dict.values()]
    final_stds = [np.array(m.std_cumulative_regret)[-1] for m in metrics_dict.values()]
    x_pos = np.arange(len(algo_names))
    ax.bar(x_pos, final_regrets, yerr=final_stds, capsize=5, alpha=0.8)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Final Cumulative Regret")
    ax.set_title("Final Regret Comparison", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_names, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.995)

    fig.tight_layout()
    return fig
