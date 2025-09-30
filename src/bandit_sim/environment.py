"""MLX-based multi-armed bandit environment abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import mlx.core as mx

DistributionKind = Literal["bernoulli", "gaussian"]


def _to_float_array(values: mx.array | float | list[float], *, name: str, num_arms: int) -> mx.array:
    """Convert values into an MLX float32 array with shape ``(num_arms,)``.

    Args:
        values: Input scalar or sequence convertible to an MLX array.
        name: Friendly identifier used in validation errors.
        num_arms: Target number of arms to validate the array length against.

    Returns:
        MLX array with dtype ``float32`` and shape ``(num_arms,)``.
    """
    array = mx.array(values, dtype=mx.float32)
    if array.ndim == 0:
        array = mx.broadcast_to(array, (num_arms,))
    if array.shape != (num_arms,):
        raise ValueError(f"{name} must have shape ({num_arms},), got {array.shape}.")
    return array


@dataclass(frozen=True)
class BernoulliParams:
    """Immutable bundle for Bernoulli reward parameters."""

    probs: mx.array


@dataclass(frozen=True)
class GaussianParams:
    """Immutable bundle for Gaussian reward parameters."""

    means: mx.array
    stds: mx.array


class BanditEnvironment:
    """Multi-armed bandit environment powered by MLX arrays.

    The environment encapsulates the ""ground truth"" reward-generating process for
    each arm. Agents interact with the environment by selecting arms; rewards are
    sampled using MLX's random primitives to enable efficient vectorized
    simulation across many parallel runs.

    Args:
        num_arms: Number of slot-machine arms available in the environment.
        distribution: Reward distribution family to use ("bernoulli" or "gaussian").
        params: Dictionary containing distribution-specific parameters. See
            :meth:`BanditEnvironment.from_parameters` for details.
    """

    def __init__(
        self,
        num_arms: int,
        *,
        distribution: DistributionKind = "bernoulli",
        params: Optional[Dict[str, mx.array | float | list[float]]] = None,
    ) -> None:
        if num_arms <= 1:
            raise ValueError("BanditEnvironment requires at least two arms.")
        self._num_arms = num_arms
        self._distribution = distribution
        self._params = self._initialize_params(params)

    @classmethod
    def from_parameters(
        cls,
        *,
        bernoulli_probs: Optional[mx.array | float | list[float]] = None,
        gaussian_means: Optional[mx.array | float | list[float]] = None,
        gaussian_stds: Optional[mx.array | float | list[float]] = None,
    ) -> "BanditEnvironment":
        """Construct an environment by specifying distribution parameters explicitly.

        One of the parameter groups must be provided:

        - Bernoulli: supply ``bernoulli_probs``.
        - Gaussian: supply both ``gaussian_means`` and ``gaussian_stds``.
        """

        if bernoulli_probs is not None:
            probs = mx.array(bernoulli_probs)
            num_arms = probs.size
            return cls(
                num_arms,
                distribution="bernoulli",
                params={"probs": probs},
            )

        if gaussian_means is not None and gaussian_stds is not None:
            means = mx.array(gaussian_means)
            num_arms = means.size
            return cls(
                num_arms,
                distribution="gaussian",
                params={"means": means, "stds": gaussian_stds},
            )

        raise ValueError(
            "Provide Bernoulli probabilities or Gaussian means and standard deviations."
        )

    @property
    def num_arms(self) -> int:
        """Number of arms available in the environment."""

        return self._num_arms

    @property
    def distribution(self) -> DistributionKind:
        """Distribution family used for sampling rewards."""

        return self._distribution

    def expected_rewards(self) -> mx.array:
        """Return the expected reward for each arm as an MLX array."""

        if isinstance(self._params, BernoulliParams):
            return self._params.probs

        if isinstance(self._params, GaussianParams):
            return self._params.means

        raise RuntimeError("Unsupported distribution parameters.")

    def sample(
        self,
        key: mx.array,
        arm_indices: mx.array | int | list[int],
    ) -> Tuple[mx.array, mx.array]:
        """Sample rewards for the provided arms.

        Args:
            key: MLX random key. A new key is returned alongside the rewards.
            arm_indices: Integer array of arm selections. Supports arbitrary batch
                shapes (e.g. ``(num_simulations,)``).

        Returns:
            A tuple of ``(rewards, new_key)`` where ``rewards`` has the same shape
            as ``arm_indices`` and contains float32 samples.
        """

        arm_indices = mx.array(arm_indices, dtype=mx.int32)

        if isinstance(self._params, BernoulliParams):
            probs = self._params.probs[arm_indices]
            key, subkey = mx.random.split(key)
            rewards = mx.random.bernoulli(p=probs, key=subkey).astype(mx.float32)
            return rewards, key

        if isinstance(self._params, GaussianParams):
            means = self._params.means[arm_indices]
            stds = self._params.stds[arm_indices]
            key, subkey = mx.random.split(key)
            rewards = mx.random.normal(
                shape=means.shape,
                loc=means,
                scale=stds,
                key=subkey,
            )
            return rewards.astype(mx.float32), key

        raise RuntimeError("Unsupported distribution parameters.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_params(self, params: Optional[Dict[str, mx.array | float | list[float]]]):
        if self._distribution == "bernoulli":
            if not params or "probs" not in params:
                raise ValueError("Bernoulli distribution requires 'probs'.")
            probs = _to_float_array(params["probs"], name="probs", num_arms=self._num_arms)
            if mx.any((probs < 0.0) | (probs > 1.0)):
                raise ValueError("Bernoulli probabilities must be in [0, 1].")
            return BernoulliParams(probs=probs)

        if self._distribution == "gaussian":
            if not params or not {"means", "stds"}.issubset(params):
                raise ValueError("Gaussian distribution requires 'means' and 'stds'.")
            means = _to_float_array(params["means"], name="means", num_arms=self._num_arms)
            stds = _to_float_array(params["stds"], name="stds", num_arms=self._num_arms)
            if mx.any(stds <= 0):
                raise ValueError("Gaussian standard deviations must be positive.")
            return GaussianParams(means=means, stds=stds)

        raise ValueError(f"Unsupported distribution '{self._distribution}'.")
