"""Smoke tests for the MLX-powered bandit environment."""

import pytest

import mlx.core as mx

from bandit_sim.environment import BanditEnvironment


def test_expected_rewards_matches_bernoulli_probs() -> None:
    probs = [0.2, 0.5, 0.8]
    env = BanditEnvironment(num_arms=3, distribution="bernoulli", params={"probs": probs})
    expected = env.expected_rewards()
    mx.eval(expected)
    assert expected.shape == (3,)
    assert expected.dtype == mx.float32
    for value, target in zip(expected.tolist(), probs):
        assert value == pytest.approx(target, abs=1e-6)


def test_sample_returns_float_rewards_and_new_key() -> None:
    probs = [0.1, 0.9]
    env = BanditEnvironment(num_arms=2, distribution="bernoulli", params={"probs": probs})
    key = mx.random.key(0)
    arms = mx.array([0, 1, 0, 1], dtype=mx.int32)
    rewards, new_key = env.sample(key, arms)
    mx.eval(rewards)
    assert rewards.shape == arms.shape
    assert rewards.dtype == mx.float32
    assert new_key.shape == key.shape
    assert not mx.all(key == new_key)
    assert mx.all((rewards == 0.0) | (rewards == 1.0))


def test_gaussian_sampling_respects_shapes() -> None:
    means = [0.0, 1.0, -0.5]
    stds = [0.1, 0.2, 0.3]
    env = BanditEnvironment(
        num_arms=3,
        distribution="gaussian",
        params={"means": means, "stds": stds},
    )
    key = mx.random.key(123)
    arms = mx.array([0, 1, 2, 1], dtype=mx.int32)
    rewards, new_key = env.sample(key, arms)
    mx.eval(rewards)
    assert rewards.shape == arms.shape
    assert rewards.dtype == mx.float32
    assert new_key.shape == key.shape
