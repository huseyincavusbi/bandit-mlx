"""Thompson Sampling algorithm for Bernoulli rewards using MLX arrays."""

from __future__ import annotations

import mlx.core as mx


def _sample_gamma_shape_ge_one(
    key: mx.array, alpha: mx.array
) -> tuple[mx.array, mx.array]:
    """Sample Gamma(alpha, 1) for alpha >= 1 using Marsaglia-Tsang."""

    alpha = mx.array(alpha, dtype=mx.float32)
    d = alpha - mx.array(1.0 / 3.0, dtype=mx.float32)
    c = 1.0 / mx.sqrt(9.0 * d)

    samples = mx.zeros(alpha.shape, dtype=mx.float32)
    done = mx.zeros(alpha.shape, dtype=mx.bool_)

    while not bool(mx.all(done).item()):
        key, normal_key = mx.random.split(key)
        x = mx.random.normal(shape=alpha.shape, key=normal_key)
        key, uniform_key = mx.random.split(key)
        u = mx.random.uniform(shape=alpha.shape, key=uniform_key)

        v = (1.0 + c * x) ** 3
        is_positive = v > 0.0
        v_pos = mx.where(is_positive, v, 1.0)
        x2 = x * x
        x4 = x2 * x2
        log_u = mx.log(u)
        accept_region = (u < 1.0 - 0.0331 * x4) | (
            log_u < 0.5 * x2 + d * (1.0 - v_pos + mx.log(v_pos))
        )
        accept = (~done) & is_positive & accept_region

        value = d * v_pos
        samples = mx.where(accept, value, samples)
        done = done | accept

    return samples, key


def _sample_gamma(key: mx.array, alpha: mx.array) -> tuple[mx.array, mx.array]:
    """Vectorized Gamma sampling for arbitrary positive alpha."""

    alpha = mx.array(alpha, dtype=mx.float32)
    alpha_ge_one = alpha >= 1.0
    adjusted_alpha = mx.where(alpha_ge_one, alpha, alpha + 1.0)

    gamma_samples, key = _sample_gamma_shape_ge_one(key, adjusted_alpha)

    if bool(mx.any(~alpha_ge_one).item()):
        key, u_key = mx.random.split(key)
        u = mx.random.uniform(shape=alpha.shape, key=u_key)
        power = mx.power(u, 1.0 / alpha)
        gamma_samples = mx.where(alpha_ge_one, gamma_samples, gamma_samples * power)

    return gamma_samples, key


def _sample_beta(
    key: mx.array, alpha: mx.array, beta: mx.array
) -> tuple[mx.array, mx.array]:
    """Sample from Beta distribution using gamma samples."""

    gamma_alpha, key = _sample_gamma(key, alpha)
    gamma_beta, key = _sample_gamma(key, beta)
    total = gamma_alpha + gamma_beta
    samples = gamma_alpha / total
    return samples, key

from .base import BanditAlgorithm, BanditAlgorithmState


class ThompsonSampling(BanditAlgorithm):
    """Bayesian Thompson Sampling with Beta posteriors per arm."""

    def __init__(self, num_arms: int, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(num_arms)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Alpha and beta must be positive.")
        self._alpha = float(alpha)
        self._beta = float(beta)

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
        alpha_post = state.rewards + self._alpha
        beta_post = (state.counts.astype(mx.float32) - state.rewards) + self._beta
        samples, key = _sample_beta(key, alpha_post, beta_post)
        chosen = mx.argmax(samples, axis=1)
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
            raise ValueError("chosen_arm and reward length mismatch.")
        if state.counts.shape[0] != chosen_arm.shape[0]:
            raise ValueError("State batch dimension mismatch.")

        num_simulations = chosen_arm.shape[0]
        arms = mx.arange(self.num_arms, dtype=mx.int32)
        selection = mx.equal(mx.expand_dims(chosen_arm, axis=1), arms)
        selection_int = selection.astype(state.counts.dtype)
        selection_float = selection.astype(mx.float32)

        new_counts = state.counts + selection_int
        new_rewards = state.rewards + selection_float * mx.expand_dims(reward, axis=1)
        return BanditAlgorithmState(counts=new_counts, rewards=new_rewards)
