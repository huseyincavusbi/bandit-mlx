"""Upper Confidence Bound (UCB1) algorithm implemented with MLX arrays."""

from __future__ import annotations

import mlx.core as mx

from .base import BanditAlgorithm, BanditAlgorithmState


class UCB1(BanditAlgorithm):
    """Classic UCB1 strategy balancing exploitation and exploration."""

    def __init__(self, num_arms: int, exploration_coef: float = 2.0) -> None:
        super().__init__(num_arms)
        if exploration_coef <= 0.0:
            raise ValueError("exploration_coef must be positive.")
        self._coef = float(exploration_coef)

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
        counts = state.counts.astype(mx.float32)
        total_counts = mx.sum(counts, axis=1)
        zero_mask = state.counts == 0
        zero_exists = mx.any(zero_mask, axis=1)
        zero_indices = mx.argmax(zero_mask.astype(mx.int32), axis=1)

        counts_safe = mx.maximum(counts, 1.0)
        avg_rewards = state.rewards / counts_safe
        log_term = mx.log(total_counts[:, None] + 1.0)
        bonus = mx.sqrt((self._coef * log_term) / counts_safe)
        ucb_score = avg_rewards + bonus
        exploit_arms = mx.argmax(ucb_score, axis=1)

        chosen = mx.where(zero_exists, zero_indices, exploit_arms)
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
            raise ValueError("chosen_arm must be 1D.")
        if reward.ndim != 1:
            raise ValueError("reward must be 1D.")
        if chosen_arm.shape[0] != reward.shape[0]:
            raise ValueError("chosen_arm and reward must align in length.")
        if state.counts.shape[0] != chosen_arm.shape[0]:
            raise ValueError("State batch dimension mismatch.")

        num_arms = self.num_arms
        arms = mx.arange(num_arms, dtype=mx.int32)
        selection = mx.equal(mx.expand_dims(chosen_arm, axis=1), arms)
        selection_int = selection.astype(state.counts.dtype)
        selection_float = selection.astype(mx.float32)

        new_counts = state.counts + selection_int
        new_rewards = state.rewards + selection_float * mx.expand_dims(reward, axis=1)
        return BanditAlgorithmState(counts=new_counts, rewards=new_rewards)
