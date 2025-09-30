"""Scaffolding for running multi-armed bandit simulations with MLX."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from bandit_sim.algorithms import BanditAlgorithm
from bandit_sim.environment import BanditEnvironment


@dataclass
class SimulationConfig:
    """Configuration bundle for multi-run simulations."""

    horizon: int
    runs: int


@dataclass
class SimulationResults:
    """Container for all outputs from a multi-run bandit simulation."""

    cumulative_rewards: mx.array
    cumulative_regret: mx.array
    arm_selections: mx.array
    rewards_per_step: mx.array
    regret_per_step: mx.array


class BanditSimulation:
    """Coordinate interactions between an environment and an algorithm."""

    def __init__(
        self,
        environment: BanditEnvironment,
        algorithm: BanditAlgorithm,
        config: SimulationConfig,
    ) -> None:
        if algorithm.num_arms != environment.num_arms:
            raise ValueError("Algorithm and environment must agree on number of arms.")
        self._env = environment
        self._algo = algorithm
        self._config = config

    @property
    def environment(self) -> BanditEnvironment:
        return self._env

    @property
    def algorithm(self) -> BanditAlgorithm:
        return self._algo

    @property
    def config(self) -> SimulationConfig:
        return self._config

    def run(self, key: mx.array) -> SimulationResults:
        """Execute bandit simulation across multiple parallel runs.

        Args:
            key: MLX random key for reproducible randomness.

        Returns:
            SimulationResults containing cumulative metrics and per-step data.
        """
        horizon = self._config.horizon
        runs = self._config.runs

        # Initialize algorithm state for all parallel runs
        state = self._algo.initialize(num_simulations=runs)

        # Pre-compute optimal arm and expected rewards for regret calculation
        expected_rewards = self._env.expected_rewards()
        optimal_arm = mx.argmax(expected_rewards)
        optimal_reward = expected_rewards[optimal_arm]

        # Storage for per-step tracking
        rewards_history = mx.zeros((runs, horizon), dtype=mx.float32)
        regret_history = mx.zeros((runs, horizon), dtype=mx.float32)
        arm_selections_history = mx.zeros((runs, horizon), dtype=mx.int32)

        # Main simulation loop over time steps
        for t in range(horizon):
            # Select arms for all runs
            chosen_arms, key = self._algo.select_arm(state, key)

            # Sample rewards from environment
            rewards, key = self._env.sample(key, chosen_arms)

            # Compute instantaneous regret based on expected reward of chosen arm
            # Regret = E[optimal] - E[chosen], NOT optimal - actual_reward
            chosen_expected_rewards = expected_rewards[chosen_arms]
            instant_regret = optimal_reward - chosen_expected_rewards

            # Store step data
            rewards_history[:, t] = rewards
            regret_history[:, t] = instant_regret
            arm_selections_history[:, t] = chosen_arms

            # Update algorithm state
            state = self._algo.update_state(state, chosen_arms, rewards)

        # Compute cumulative metrics
        cumulative_rewards = mx.cumsum(rewards_history, axis=1)
        cumulative_regret = mx.cumsum(regret_history, axis=1)

        return SimulationResults(
            cumulative_rewards=cumulative_rewards,
            cumulative_regret=cumulative_regret,
            arm_selections=arm_selections_history,
            rewards_per_step=rewards_history,
            regret_per_step=regret_history,
        )
