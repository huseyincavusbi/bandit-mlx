"""Abstract interfaces for bandit algorithms using MLX state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BanditAlgorithmState:
    """Container for tracking algorithm-specific statistics per arm."""

    counts: mx.array
    rewards: mx.array


class BanditAlgorithm(ABC):
    """Base class for all bandit selection strategies."""

    def __init__(self, num_arms: int) -> None:
        if num_arms < 1:
            raise ValueError("BanditAlgorithm requires at least one arm.")
        self._num_arms = num_arms

    @property
    def num_arms(self) -> int:
        return self._num_arms

    @abstractmethod
    def initialize(self, num_simulations: int) -> BanditAlgorithmState:
        """Create algorithm state tensors stored as MLX arrays.

        Args:
            num_simulations: Number of independent bandit runs to maintain.
        """

    @abstractmethod
    def select_arm(
        self,
        state: BanditAlgorithmState,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Return chosen arm index per simulation alongside the next random key."""

    @abstractmethod
    def update_state(
        self,
        state: BanditAlgorithmState,
        chosen_arm: mx.array,
        reward: mx.array,
    ) -> BanditAlgorithmState:
        """Update internal statistics given observed reward.

        Args:
            state: Current algorithm state.
            chosen_arm: Integer array of shape ``(num_simulations,)`` identifying
                which arm was pulled in each simulation.
            reward: Float array of shape ``(num_simulations,)`` describing
                observed rewards.
        """
