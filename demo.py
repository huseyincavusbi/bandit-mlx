"""Main demonstration script comparing all three bandit algorithms."""

from __future__ import annotations

import mlx.core as mx

from bandit_sim import BanditEnvironment, BanditSimulation, SimulationConfig
from bandit_sim.algorithms import EpsilonGreedy, ThompsonSampling, UCB1
from bandit_sim.metrics import compare_algorithms
from bandit_sim.visualization import create_comparison_dashboard

import matplotlib.pyplot as plt


def run_easy_scenario():
    """Easy scenario: One arm is clearly better than others."""
    print("\n" + "="*70)
    print("EASY SCENARIO: One clearly superior arm")
    print("="*70)
    
    # 10-armed bandit with one arm clearly best (prob=0.9)
    probs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.9]
    print(f"Arm probabilities: {probs}")
    print(f"Optimal arm: 9 (prob={probs[9]})")
    
    env = BanditEnvironment(num_arms=10, distribution="bernoulli", params={"probs": probs})
    config = SimulationConfig(horizon=1000, runs=100)
    
    # Initialize algorithms
    epsilon_greedy = EpsilonGreedy(num_arms=10, epsilon=0.1)
    ucb = UCB1(num_arms=10, exploration_coef=2.0)
    thompson = ThompsonSampling(num_arms=10, alpha=1.0, beta=1.0)
    
    # Run simulations
    print("\nRunning simulations with 100 independent runs over 1000 timesteps...")
    key = mx.random.key(42)
    
    results = {}
    for name, algo in [
        ("ε-Greedy (ε=0.1)", epsilon_greedy),
        ("UCB1 (c=2.0)", ucb),
        ("Thompson Sampling", thompson),
    ]:
        print(f"  - {name}")
        sim = BanditSimulation(environment=env, algorithm=algo, config=config)
        key, subkey = mx.random.split(key)
        results[name] = sim.run(subkey)
    
    # Compute metrics
    print("\nComputing aggregate metrics...")
    metrics = compare_algorithms(results)
    
    # Print final results
    print("\nFinal Results (mean ± std):")
    for name, metric in metrics.items():
        final_regret_mean = float(metric.mean_cumulative_regret[-1].item())
        final_regret_std = float(metric.std_cumulative_regret[-1].item())
        final_reward_mean = float(metric.mean_cumulative_reward[-1].item())
        final_reward_std = float(metric.std_cumulative_reward[-1].item())
        print(f"  {name}:")
        print(f"    Cumulative Reward: {final_reward_mean:.2f} ± {final_reward_std:.2f}")
        print(f"    Cumulative Regret: {final_regret_mean:.2f} ± {final_regret_std:.2f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = create_comparison_dashboard(
        metrics,
        num_arms=10,
        suptitle="Easy Scenario: 10-Armed Bandit (One Clear Winner)",
    )
    plt.savefig("easy_scenario.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: easy_scenario.png")
    
    return results, metrics


def run_hard_scenario():
    """Hard scenario: Multiple arms with similar rewards."""
    print("\n" + "="*70)
    print("HARD SCENARIO: Arms with similar rewards")
    print("="*70)
    
    # 10-armed bandit with arms clustered around similar values
    probs = [0.45, 0.48, 0.50, 0.52, 0.55, 0.53, 0.51, 0.49, 0.47, 0.54]
    print(f"Arm probabilities: {probs}")
    print(f"Optimal arm: 4 (prob={probs[4]})")
    
    env = BanditEnvironment(num_arms=10, distribution="bernoulli", params={"probs": probs})
    config = SimulationConfig(horizon=1000, runs=100)
    
    # Initialize algorithms
    epsilon_greedy = EpsilonGreedy(num_arms=10, epsilon=0.1)
    ucb = UCB1(num_arms=10, exploration_coef=2.0)
    thompson = ThompsonSampling(num_arms=10, alpha=1.0, beta=1.0)
    
    # Run simulations
    print("\nRunning simulations with 100 independent runs over 1000 timesteps...")
    key = mx.random.key(123)
    
    results = {}
    for name, algo in [
        ("ε-Greedy (ε=0.1)", epsilon_greedy),
        ("UCB1 (c=2.0)", ucb),
        ("Thompson Sampling", thompson),
    ]:
        print(f"  - {name}")
        sim = BanditSimulation(environment=env, algorithm=algo, config=config)
        key, subkey = mx.random.split(key)
        results[name] = sim.run(subkey)
    
    # Compute metrics
    print("\nComputing aggregate metrics...")
    metrics = compare_algorithms(results)
    
    # Print final results
    print("\nFinal Results (mean ± std):")
    for name, metric in metrics.items():
        final_regret_mean = float(metric.mean_cumulative_regret[-1].item())
        final_regret_std = float(metric.std_cumulative_regret[-1].item())
        final_reward_mean = float(metric.mean_cumulative_reward[-1].item())
        final_reward_std = float(metric.std_cumulative_reward[-1].item())
        print(f"  {name}:")
        print(f"    Cumulative Reward: {final_reward_mean:.2f} ± {final_reward_std:.2f}")
        print(f"    Cumulative Regret: {final_regret_mean:.2f} ± {final_regret_std:.2f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = create_comparison_dashboard(
        metrics,
        num_arms=10,
        suptitle="Hard Scenario: 10-Armed Bandit (Similar Rewards)",
    )
    plt.savefig("hard_scenario.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: hard_scenario.png")
    
    return results, metrics


def main():
    """Run both scenarios and display results."""
    print("\n" + "#"*70)
    print("# Multi-Armed Bandit Simulator - MLX Implementation")
    print("#"*70)
    
    # Run easy scenario
    easy_results, easy_metrics = run_easy_scenario()
    
    # Run hard scenario
    hard_results, hard_metrics = run_hard_scenario()
    
    print("\n" + "="*70)
    print("Simulation complete! Check the generated PNG files for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
