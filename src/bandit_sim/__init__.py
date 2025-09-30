"""Multi-Armed Bandit Simulator with MLX."""

from .environment import BanditEnvironment
from .algorithms.base import BanditAlgorithm, BanditAlgorithmState
from .algorithms.epsilon_greedy import EpsilonGreedy
from .algorithms.ucb import UCB1
from .algorithms.thompson import ThompsonSampling
from .simulation.runner import BanditSimulation, SimulationConfig, SimulationResults
from .metrics import AggregatedMetrics, compute_metrics, compare_algorithms
from .visualization import (
    plot_cumulative_regret,
    plot_cumulative_reward,
    plot_arm_frequencies,
    create_comparison_dashboard,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "BanditEnvironment",
    "BanditAlgorithm",
    "BanditAlgorithmState",
    # Algorithms
    "EpsilonGreedy",
    "UCB1",
    "ThompsonSampling",
    # Simulation
    "BanditSimulation",
    "SimulationConfig",
    "SimulationResults",
    # Metrics
    "AggregatedMetrics",
    "compute_metrics",
    "compare_algorithms",
    # Visualization
    "plot_cumulative_regret",
    "plot_cumulative_reward",
    "plot_arm_frequencies",
    "create_comparison_dashboard",
]