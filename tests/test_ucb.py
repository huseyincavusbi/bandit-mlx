"""Unit tests for the MLX-based UCB1 algorithm."""

import mlx.core as mx

from bandit_sim.algorithms import BanditAlgorithmState, UCB1


def test_initialize_zero_state() -> None:
    algo = UCB1(num_arms=4, exploration_coef=2.0)
    state = algo.initialize(num_simulations=3)
    assert state.counts.shape == (3, 4)
    assert state.rewards.shape == (3, 4)
    assert bool(mx.all(state.counts == 0).item())
    assert bool(mx.all(state.rewards == 0.0).item())


def test_select_prioritizes_unpulled_arms() -> None:
    algo = UCB1(num_arms=3)
    counts = mx.array([[1, 0, 5], [2, 3, 4]], dtype=mx.int32)
    rewards = mx.array([[1.0, 0.0, 2.0], [2.0, 4.0, 6.0]], dtype=mx.float32)
    state = BanditAlgorithmState(counts=counts, rewards=rewards)
    key = mx.random.key(0)
    chosen, new_key = algo.select_arm(state, key)
    assert chosen.tolist() == [1, 2]
    assert bool(mx.all(key == new_key).item())


def test_select_uses_confidence_bonus() -> None:
    algo = UCB1(num_arms=2, exploration_coef=2.0)
    counts = mx.array([[10, 2]], dtype=mx.int32)
    rewards = mx.array([[8.0, 1.8]], dtype=mx.float32)
    state = BanditAlgorithmState(counts=counts, rewards=rewards)
    key = mx.random.key(42)
    chosen, _ = algo.select_arm(state, key)
    # Arm 0 average is 0.8, arm 1 average is 0.9, but arm 1 has few pulls so should be selected.
    assert chosen.tolist() == [1]


def test_update_state_accumulates_counts_and_rewards() -> None:
    algo = UCB1(num_arms=3)
    state = algo.initialize(num_simulations=2)
    chosen = mx.array([0, 2], dtype=mx.int32)
    rewards = mx.array([1.0, 0.5], dtype=mx.float32)
    state = algo.update_state(state, chosen, rewards)
    assert state.counts.tolist() == [[1, 0, 0], [0, 0, 1]]
    assert state.rewards.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.5]]