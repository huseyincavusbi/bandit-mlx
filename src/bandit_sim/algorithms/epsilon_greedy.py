"""Epsilon-greedy bandit algorithm implemented with MLX arrays."""

from __future__ import annotations

import mlx.core as mx

from .base import BanditAlgorithm, BanditAlgorithmState

class EpsilonGreedy(BanditAlgorithm):
    """Select arms using an epsilon-greedy exploration policy."""

    def __init__(self, num_arms: int, epsilon: float) -> None:
        super().__init__(num_arms)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must lie in [0, 1].")
        self._epsilon = float(epsilon)

    def initialize(self, num_simulations: int) -> BanditAlgorithmState:
        if num_simulations < 1:
            raise ValueError("num_simulations must be positive.")
        counts = mx.zeros((num_simulations, self.num_arms), dtype=mx.int32)
        rewards = mx.zeros((num_simulations, self.num_arms), dtype=mx.float32)
        return BanditAlgorithmState(counts=counts, rewards=rewards)

    def select_arm(
        self,
        state: BanditAlgorithmState,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        num_simulations = state.counts.shape[0]
        counts_float = state.counts.astype(mx.float32)
        safe_counts = mx.maximum(counts_float, 1.0)
        avg_rewards = state.rewards / safe_counts

        exploit_arms = mx.argmax(avg_rewards, axis=1)
        key, explore_key = mx.random.split(key)
        explore_draw = mx.random.uniform(shape=(num_simulations,), key=explore_key)
        explore_mask = explore_draw < self._epsilon
        key, random_key = mx.random.split(key)
        random_arms = mx.random.randint(
            low=0,
            high=self.num_arms,
            shape=(num_simulations,),
            key=random_key,
        )
        chosen = mx.where(explore_mask, random_arms, exploit_arms)
        return chosen.astype(mx.int32), key

    def update_state(
        self,
        state: BanditAlgorithmState,
        chosen_arm: mx.array,
        reward: mx.array,
    ) -> BanditAlgorithmState:
        chosen_arm = mx.array(chosen_arm, dtype=mx.int32)
        reward = mx.array(reward, dtype=mx.float32)

        if chosen_arm.ndim != 1:
            raise ValueError("chosen_arm must be a 1D array of arm indices.")
        if reward.ndim != 1:
            raise ValueError("reward must be a 1D array of rewards.")
        if chosen_arm.shape[0] != reward.shape[0]:
            raise ValueError("chosen_arm and reward must have matching length.")

        num_simulations = chosen_arm.shape[0]
        if state.counts.shape[0] != num_simulations:
            raise ValueError("State and updates disagree on num_simulations.")

        arms = mx.arange(self.num_arms, dtype=mx.int32)
        one_hot = mx.equal(mx.expand_dims(chosen_arm, axis=1), arms)
        one_hot_int = one_hot.astype(state.counts.dtype)
        one_hot_float = one_hot.astype(mx.float32)

        updated_counts = state.counts + one_hot_int
        updated_rewards = state.rewards + one_hot_float * mx.expand_dims(reward, axis=1)

        return BanditAlgorithmState(counts=updated_counts, rewards=updated_rewards)
