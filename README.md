# Multi-Armed Bandit Simulator with MLX

A high-performance multi-armed bandit simulator built on Apple's MLX framework, featuring parallel simulations and comprehensive algorithm comparisons.

## Features

- **MLX-Powered**: Leverages Apple's MLX framework for efficient array operations on Apple Silicon
- **Multiple Algorithms**: 
  - ε-Greedy (Epsilon-Greedy)
  - UCB1 (Upper Confidence Bound)
  - Thompson Sampling (Bayesian approach)
- **Parallel Simulations**: Run hundreds of parallel simulations efficiently
- **Multiple Distributions**: Support for Bernoulli and Gaussian reward distributions
- **Rich Metrics**: Track cumulative rewards, regret, and arm selection frequencies
- **Visualization**: Built-in plotting functions for algorithm comparison

## Installation

This project uses a virtual environment. Activate it using `uv`:

```bash
# From the project root
cd /path/to/bandit
source bandit/bin/activate
```

Install dependencies:

```bash
uv pip install mlx matplotlib pytest
```

## Quick Start

### Basic Example

```python
import mlx.core as mx
from bandit_sim import (
    BanditEnvironment,
    EpsilonGreedy,
    BanditSimulation,
    SimulationConfig
)

# Create a 5-arm Bernoulli bandit
env = BanditEnvironment(
    num_arms=5,
    distribution="bernoulli",
    params={"probs": [0.1, 0.3, 0.5, 0.7, 0.9]}  # Arm probabilities
)

# Set up an epsilon-greedy algorithm
algorithm = EpsilonGreedy(num_arms=5, epsilon=0.1)

# Configure simulation
config = SimulationConfig(horizon=1000, runs=100)

# Run simulation
simulation = BanditSimulation(env, algorithm, config)
key = mx.random.key(42)
results = simulation.run(key)

print(f"Final cumulative reward: {results.cumulative_rewards[:, -1].mean():.2f}")
print(f"Final cumulative regret: {results.cumulative_regret[:, -1].mean():.2f}")
```

### Running the Demo

The demo compares all three algorithms on two scenarios:

```bash
cd /path/to/bandit
PYTHONPATH=./bandit/src uv run --python ./bandit/bin/python python ./bandit/demo.py
```

This generates comparison visualizations saved as PNG files.

## Project Structure

```
bandit/
├── src/bandit_sim/          # Main package
│   ├── __init__.py          # Package exports
│   ├── environment.py       # Bandit environment
│   ├── algorithms/          # Algorithm implementations
│   │   ├── __init__.py     # Algorithm exports
│   │   ├── base.py         # Abstract base class
│   │   ├── epsilon_greedy.py
│   │   ├── ucb.py
│   │   └── thompson.py
│   ├── simulation/          # Simulation engine
│   │   ├── __init__.py     # Simulation exports
│   │   └── runner.py
│   ├── metrics.py           # Metric computation
│   └── visualization.py     # Plotting functions
├── notebooks/               # Introductory Jupyter notebooks
│   ├── introduction.ipynb   # Basic bandit concepts and algorithms
│   └── clinical_trial_example.ipynb  # Real-world clinical trial application
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py          
│   ├── test_environment.py
│   ├── test_epsilon_greedy.py
│   ├── test_ucb.py
│   ├── test_thompson.py
│   └── test_simulation.py
├── demo.py                  # Demonstration script
├── README.md                # This file
├── DESIGN.md                # Implementation details
├── requirements.txt         # Package dependencies
└── .gitignore               # Git ignore rules
```

**Note**: Virtual environment directories (`bin/`, `lib/`, `share/`, etc.) and generated output files (`.png` visualizations) are not shown above.

## Algorithm Descriptions

### ε-Greedy (Epsilon-Greedy)

Balances exploration and exploitation using a simple random strategy:
- With probability ε: explore (random arm)
- With probability 1-ε: exploit (best arm so far)

**Parameters:**
- `epsilon` (float): Exploration probability (default: 0.1)

### UCB1 (Upper Confidence Bound)

Selects arms based on optimistic estimates with confidence intervals:

```
UCB(arm) = mean_reward + c * sqrt(log(t) / n_arm)
```

Where `c` controls exploration strength.

**Parameters:**
- `exploration_coef` (float): Exploration coefficient (default: 2.0)

### Thompson Sampling

Bayesian approach that samples from posterior distributions:
- Maintains Beta distributions for each arm
- Samples from posteriors and selects argmax
- Updates posteriors based on observed rewards

**Implementation Note:** MLX doesn't provide a native Beta distribution, so we implement it using a custom Gamma sampler (Marsaglia-Tsang algorithm).

## Design and Implementation

For detailed information about MLX-specific design decisions, implementation patterns, and architectural choices, see **[DESIGN.md](DESIGN.md)**.

Key topics covered:
- Functional random number generation with key threading
- Vectorized parallel execution patterns
- Custom Beta distribution implementation
- Expected vs. actual regret calculation
- One-hot encoding for efficient updates
- Array shapes and broadcasting conventions

## Metrics

### Cumulative Reward
Sum of all rewards received up to each timestep.

### Cumulative Regret
The difference between the optimal expected reward and the expected reward of chosen arms:

```
Regret(t) = t * max_reward - sum(expected_rewards[chosen_arms])
```

**Note:** We use expected rewards (not actual sampled rewards) to ensure monotonic regret and fair algorithm comparison.

### Arm Selection Frequencies
Count of how often each arm was selected across all runs.

## Visualization

### Available Plots

1. **Cumulative Regret**: Line plot with standard deviation bands
2. **Cumulative Reward**: Line plot with standard deviation bands  
3. **Arm Frequencies**: Bar chart of selection counts
4. **Comparison Dashboard**: 2×2 grid combining all metrics

### Example

```python
import matplotlib.pyplot as plt
from bandit_sim import create_comparison_dashboard, compare_algorithms

# Compare multiple algorithms
results_dict = {
    "ε-Greedy": results_eg,
    "UCB1": results_ucb,
    "Thompson": results_ts,
}

metrics_dict = compare_algorithms(results_dict)
fig = create_comparison_dashboard(metrics_dict, num_arms=5)
fig.savefig("comparison.png", dpi=150, bbox_inches="tight")
```

## Running Tests

```bash
cd /path/to/bandit
PYTHONPATH=./bandit/src uv run --python ./bandit/bin/python pytest bandit/tests/ -v
```

All tests should pass:
- `test_environment.py`: Environment sampling and expected rewards
- `test_epsilon_greedy.py`: ε-greedy selection and updates
- `test_ucb.py`: UCB confidence bounds and unpulled arm prioritization
- `test_thompson.py`: Beta sampling and posterior updates
- `test_simulation.py`: Cumulative metrics and arm selection validity

## Performance Characteristics

### Demo Results (100 runs × 1000 timesteps)

**Easy Scenario** (one clearly superior arm):
- Thompson Sampling: **27.15 ± 6.38** regret (best)
- ε-Greedy: 119.08 ± 66.25 regret
- UCB1: 140.94 ± 10.86 regret

**Hard Scenario** (similar arm rewards):
- ε-Greedy: 31.59 ± 14.88 regret
- Thompson Sampling: 35.49 ± 6.93 regret
- UCB1: 41.15 ± 2.27 regret

Thompson Sampling generally performs best when there's a clear optimal arm, while all algorithms perform similarly when arms are close in value.

## API Reference

### Core Classes

#### `BanditEnvironment`
```python
BanditEnvironment(
    num_arms: int,
    distribution: str,  # "bernoulli" or "gaussian"
    params: list | mx.array
)
```

Methods:
- `sample(key, arm_indices)`: Generate rewards for selected arms
- `expected_rewards()`: Get mean reward for each arm

#### `BanditAlgorithm` (Abstract Base)

All algorithms implement:
- `initialize(num_simulations)`: Create initial state
- `select_arm(state, key)`: Choose arms for all simulations
- `update_state(state, arm_indices, rewards)`: Update based on observations

#### `SimulationConfig`
```python
SimulationConfig(
    horizon: int,        # Number of timesteps
    num_runs: int        # Number of parallel simulations
)
```

#### `BanditSimulation`
```python
BanditSimulation(
    environment: BanditEnvironment,
    algorithm: BanditAlgorithm,
    config: SimulationConfig
)
```

Methods:
- `run(key)`: Execute simulation and return `SimulationResults`

### Metrics Functions

- `compute_metrics(results)`: Aggregate statistics across runs
- `compare_algorithms(results_dict)`: Compute metrics for multiple algorithms

### Visualization Functions

- `plot_cumulative_regret(metrics_dict, *, title=..., xlabel=..., ylabel=..., show_std=True, figsize=(10,6))`: Plot regret curves
- `plot_cumulative_reward(metrics_dict, *, title=..., xlabel=..., ylabel=..., show_std=True, figsize=(10,6))`: Plot reward curves
- `plot_arm_frequencies(metrics_dict, num_arms, *, title=..., xlabel=..., ylabel=..., figsize=(10,6))`: Plot arm selection frequencies
- `create_comparison_dashboard(metrics_dict, num_arms, *, suptitle=None, figsize=(15,10))`: Create 2×2 comparison dashboard

All visualization functions return a `matplotlib.figure.Figure` object that can be saved with `fig.savefig()`.

## License

MIT License

## Acknowledgments

Built with [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework for Apple Silicon.
