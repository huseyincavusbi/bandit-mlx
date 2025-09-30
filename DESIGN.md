# MLX Implementation Design Notes

## Overview

This document explains key design decisions made when implementing the multi-armed bandit simulator using Apple's MLX framework.

## Core Design Principles

### 1. Functional Random Number Generation

**Decision:** Use JAX-style key splitting for reproducible randomness.

**Implementation:**
```python
# In simulation loop
for t in range(horizon):
    chosen_arms, key = self.algorithm.select_arm(state, key)
    rewards, key = self.environment.sample(key, chosen_arms)
    # key is threaded through all random operations
```

**Benefits:**
- Deterministic simulations with same seed
- Independent random streams across parallel simulations
- No hidden global state

---

### 2. Vectorized Parallel Execution

**Decision:** Process all simulations simultaneously using array operations.

**Implementation:**
```python
# Update state for ALL simulations at once
# state.counts shape: (num_runs, num_arms)
# chosen_arms shape: (num_runs,)

one_hot = mx.equal(mx.expand_dims(chosen_arm, axis=1), mx.arange(num_arms))
updated_counts = state.counts + one_hot.astype(mx.int32)
```

**Benefits:**
- Leverages MLX's optimized array operations
- No Python loops over simulations
- Scales efficiently with number of parallel runs

---

### 3. Custom Beta Distribution

**Problem:** MLX lacks `mx.random.beta()` needed for Thompson Sampling.

**Solution:** Use the mathematical identity:
```
If X ~ Gamma(α) and Y ~ Gamma(β), then X/(X+Y) ~ Beta(α, β)
```

**Implementation:**
```python
def _sample_beta(key, alpha, beta):
    gamma_alpha, key = _sample_gamma(key, alpha)
    gamma_beta, key = _sample_gamma(key, beta)
    return gamma_alpha / (gamma_alpha + gamma_beta), key
```

The Gamma sampler uses Marsaglia-Tsang algorithm with while-loop acceptance-rejection.

---

### 4. Expected vs. Actual Regret

**Decision:** Compute regret using expected rewards, not sampled rewards.

**Implementation:**
```python
# Regret = E[optimal] - E[chosen], NOT optimal - actual_reward
chosen_expected_rewards = expected_rewards[chosen_arms]
instant_regret = optimal_reward - chosen_expected_rewards
```

**Rationale:**
- Measures algorithm performance, not sampling variance
- Ensures monotonically increasing cumulative regret
- Standard practice in bandit research

---

### 5. One-Hot Encoding Pattern

**Decision:** Use broadcasting for efficient arm-specific updates.

**Implementation:**
```python
# chosen_arms: (num_runs,) - which arm each simulation chose
# Create (num_runs, num_arms) mask where chosen arms = 1, others = 0
one_hot = mx.equal(
    mx.expand_dims(chosen_arms, 1),  # (num_runs, 1)
    mx.arange(num_arms)              # (num_arms,)
)  # broadcasts to (num_runs, num_arms)

# Update rewards only for chosen arms
updated_rewards = state.rewards + one_hot * mx.expand_dims(rewards, 1)
```

---

## Implementation Notes

### Why Custom Samplers?

MLX is relatively new and lacks some distributions (Beta, advanced Gamma variants). We implement custom samplers to stay within the MLX ecosystem rather than mixing with NumPy.

### Array Shapes Convention

- First dimension: parallel simulations (`num_runs`)
- Second dimension: arms (`num_arms`)  
- Third dimension (when present): time steps (`horizon`)

Example: `cumulative_rewards` has shape `(num_runs, horizon)`

### Key Threading Pattern

Every random operation:
1. Splits the key: `key, subkey = mx.random.split(key)`
2. Uses subkey for random operation
3. Returns updated key

This ensures independent random streams and reproducibility.

---

## Validation Approach

### Code Verification
- Unit tests for each algorithm component
- Integration tests for simulation pipeline
- Edge case testing (single arm, equal rewards, extreme parameters)

### Practical Testing
- Demo scenarios with clear optimal arms
- Scenarios with similar arm values  
- Visual inspection of learning curves

---

## Key Takeaways

1. **Functional RNG**: Key splitting ensures reproducibility
2. **Vectorization**: Process all simulations simultaneously  
3. **Custom Distributions**: Implement missing samplers when needed
4. **Expected Regret**: Use expected values for fair algorithm comparison
5. **Dataclasses**: Clear structure for state and results
6. **Broadcasting**: One-hot encoding for efficient updates

This design balances performance, clarity, and maintainability while working within MLX's current capabilities.
