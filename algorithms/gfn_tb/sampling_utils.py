import warnings
from functools import partial
from typing import Callable, Literal

import chex
import jax
import jax.numpy as jnp
from jax import random


def multinomial(
    key: jax.Array, weights: chex.Array, N: int, replacement: bool = True
) -> chex.Array:
    """Return sampled indices from multinomial distribution.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
        replacement: Whether to sample with replacement.
    """
    indices = jnp.arange(len(weights))
    return random.choice(
        key,
        indices,
        shape=(N,),
        replace=replacement,
        p=jax.nn.softmax(weights),
    )


def stratified(key: jax.Array, weights: chex.Array, N: int, replacement: bool = True) -> chex.Array:
    """Return sampled indices using stratified (re)sampling technique.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
    """
    if not replacement:
        warnings.warn(
            "Stratified sampling does not support sampling without replacement. "
            "Using multinomial sampling instead."
        )
        return multinomial(key, weights, N, replacement=True)

    # Normalize weights if they're not already normalized
    weights = weights / weights.sum()

    cumsum = jnp.cumsum(weights)
    u = (jnp.arange(N) + random.uniform(key, shape=(N,))) / N
    indices = jnp.searchsorted(cumsum, u)
    return jnp.clip(indices, 0, len(weights) - 1)


def systematic(key: jax.Array, weights: chex.Array, N: int, replacement: bool = True) -> chex.Array:
    """Return sampled indices using systematic (re)sampling technique.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
    """
    if not replacement:
        warnings.warn(
            "Systematic sampling does not support sampling without replacement. "
            "Using multinomial sampling instead."
        )
        return multinomial(key, weights, N, replacement=True)

    # Normalize weights
    weights = weights / weights.sum()

    cumsum = jnp.cumsum(weights)
    u = (jnp.arange(N) + random.uniform(key, shape=(1,))) / N
    indices = jnp.searchsorted(cumsum, u)
    return jnp.clip(indices, 0, len(weights) - 1)


def rank(
    key: jax.Array,
    weights: chex.Array,
    N: int,
    replacement: bool = True,
    rank_k: float = 0.01,
) -> chex.Array:
    """Return sampled indices using rank-based (re)sampling technique.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
        replacement: Whether to sample with replacement.
        rank_k: A hyperparameter for rank-based sampling.
    """
    ranks = jnp.argsort(jnp.argsort(-weights))  # type: ignore
    new_weights = 1.0 / (rank_k * len(weights) + ranks)
    return multinomial(key, new_weights, N, replacement=replacement)


def get_sampling_func(
    sampling_strategy: Literal["multinomial", "stratified", "systematic", "rank"],
    rank_k: float = 0.01,
) -> Callable[[jax.Array, chex.Array, int, bool], chex.Array]:
    """Factory function to get the desired sampling method.

    Args:
        sampling_strategy: The name of the sampling strategy.
        rank_k: A hyperparameter for rank-based sampling, used only if strategy is 'rank'.

    Returns:
        A callable sampling function.
    """
    if sampling_strategy == "multinomial":
        return multinomial
    elif sampling_strategy == "stratified":
        return stratified
    elif sampling_strategy == "systematic":
        return systematic
    elif sampling_strategy == "rank":
        return partial(rank, rank_k=rank_k)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
