"""Unit tests for the Thompson Sampling algorithm."""

import mlx.core as mx

from bandit_sim.algorithms import BanditAlgorithmState, ThompsonSampling


def test_initialize_zero_state() -> None:
    algo = ThompsonSampling(num_arms=3, alpha=1.0, beta=1.0)
    state = algo.initialize(num_simulations=2)
    assert state.counts.shape == (2, 3)
    assert state.rewards.shape == (2, 3)
    assert bool(mx.all(state.counts == 0).item())
    assert bool(mx.all(state.rewards == 0.0).item())


def test_select_arm_samples_from_beta() -> None:
    algo = ThompsonSampling(num_arms=2, alpha=1.0, beta=1.0)
    counts = mx.array([[10, 10]], dtype=mx.int32)
    rewards = mx.array([[8.0, 2.0]], dtype=mx.float32)
    state = BanditAlgorithmState(counts=counts, rewards=rewards)
    key = mx.random.key(0)
    chosen, new_key = algo.select_arm(state, key)
    assert chosen.shape == (1,)
    assert not bool(mx.all(key == new_key).item())


def test_update_state_accumulates_posterior_counts() -> None:
    algo = ThompsonSampling(num_arms=2, alpha=1.0, beta=1.0)
    state = algo.initialize(num_simulations=1)
    chosen = mx.array([1], dtype=mx.int32)
    reward = mx.array([1.0], dtype=mx.float32)
    state = algo.update_state(state, chosen, reward)
    assert state.counts.tolist() == [[0, 1]]
    assert state.rewards.tolist() == [[0.0, 1.0]]

    chosen_second = mx.array([0], dtype=mx.int32)
    reward_second = mx.array([0.0], dtype=mx.float32)
    state = algo.update_state(state, chosen_second, reward_second)
    assert state.counts.tolist() == [[1, 1]]
    assert state.rewards.tolist() == [[0.0, 1.0]]
