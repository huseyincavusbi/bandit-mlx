"""Unit tests for the bandit simulation runner."""

import mlx.core as mx

from bandit_sim import BanditEnvironment, SimulationConfig, BanditSimulation
from bandit_sim.algorithms import EpsilonGreedy


def test_simulation_tracks_rewards_and_regret() -> None:
    env = BanditEnvironment(num_arms=3, distribution="bernoulli", params={"probs": [0.1, 0.5, 0.9]})
    algo = EpsilonGreedy(num_arms=3, epsilon=0.1)
    config = SimulationConfig(horizon=10, runs=5)
    sim = BanditSimulation(environment=env, algorithm=algo, config=config)
    
    key = mx.random.key(42)
    results = sim.run(key)
    
    assert results.cumulative_rewards.shape == (5, 10)
    assert results.cumulative_regret.shape == (5, 10)
    assert results.arm_selections.shape == (5, 10)
    assert results.rewards_per_step.shape == (5, 10)
    assert results.regret_per_step.shape == (5, 10)
    
    # Cumulative rewards should be non-decreasing (allowing small floating-point errors)
    diffs = results.cumulative_rewards[:, 1:] - results.cumulative_rewards[:, :-1]
    assert bool(mx.all(diffs >= -1e-5).item())
    
    # Cumulative regret should be non-negative and monotonically increasing
    # (regret can only increase or stay the same, never decrease)
    assert bool(mx.all(results.cumulative_regret >= 0.0).item())
    regret_diffs = results.cumulative_regret[:, 1:] - results.cumulative_regret[:, :-1]
    assert bool(mx.all(regret_diffs >= -1e-5).item())


def test_simulation_arm_selections_are_valid() -> None:
    env = BanditEnvironment(num_arms=4, distribution="bernoulli", params={"probs": [0.2, 0.3, 0.6, 0.4]})
    algo = EpsilonGreedy(num_arms=4, epsilon=0.2)
    config = SimulationConfig(horizon=20, runs=3)
    sim = BanditSimulation(environment=env, algorithm=algo, config=config)
    
    key = mx.random.key(123)
    results = sim.run(key)
    
    # All arm selections should be in valid range
    assert bool(mx.all(results.arm_selections >= 0).item())
    assert bool(mx.all(results.arm_selections < 4).item())
