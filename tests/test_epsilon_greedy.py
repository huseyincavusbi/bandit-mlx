"""Unit tests for the epsilon-greedy algorithm implementation."""

import mlx.core as mx

from bandit_sim.algorithms import BanditAlgorithmState, EpsilonGreedy


def test_initialize_creates_zero_state() -> None:
    algo = EpsilonGreedy(num_arms=3, epsilon=0.1)
    state = algo.initialize(num_simulations=5)
    assert state.counts.shape == (5, 3)
    assert state.rewards.shape == (5, 3)
    assert bool(mx.all(state.counts == 0).item())
    assert bool(mx.all(state.rewards == 0.0).item())


def test_select_arm_exploits_best_average_when_epsilon_zero() -> None:
    algo = EpsilonGreedy(num_arms=3, epsilon=0.0)
    counts = mx.array([[10, 10, 10], [5, 5, 5]], dtype=mx.int32)
    rewards = mx.array([[5.0, 8.0, 4.0], [1.0, 3.0, 5.0]], dtype=mx.float32)
    state = BanditAlgorithmState(counts=counts, rewards=rewards)
    key = mx.random.key(123)
    chosen, new_key = algo.select_arm(state, key)
    assert chosen.tolist() == [1, 2]
    same_key = bool(mx.all(key == new_key).item())
    assert not same_key


def test_update_state_accumulates_counts_and_rewards() -> None:
    algo = EpsilonGreedy(num_arms=3, epsilon=0.5)
    state = algo.initialize(num_simulations=2)
    chosen = mx.array([1, 0], dtype=mx.int32)
    reward = mx.array([1.0, 0.5], dtype=mx.float32)
    state = algo.update_state(state, chosen, reward)
    mx.eval(state.counts, state.rewards)
    assert state.counts.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert state.rewards.tolist() == [[0.0, 1.0, 0.0], [0.5, 0.0, 0.0]]

    # Apply another update and ensure accumulation.
    chosen_next = mx.array([1, 2], dtype=mx.int32)
    reward_next = mx.array([0.5, 2.0], dtype=mx.float32)
    state = algo.update_state(state, chosen_next, reward_next)
    mx.eval(state.counts, state.rewards)
    assert state.counts.tolist() == [[0, 2, 0], [1, 0, 1]]
    assert state.rewards.tolist() == [[0.0, 1.5, 0.0], [0.5, 0.0, 2.0]]
