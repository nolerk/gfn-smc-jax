import warnings
from functools import partial
from typing import Callable, Literal

import chex
import jax
import jax.numpy as jnp
from jax import random


def ess(
    log_iws: chex.Array | None = None,  # (bs,)
    normalized_weights: chex.Array | None = None,  # (bs,)
) -> chex.Array:
    if normalized_weights is None:
        assert log_iws is not None
        normalized_weights = jax.nn.softmax(log_iws, axis=0)  # (bs,)
    return 1 / (normalized_weights**2).sum()  # scalar


def binary_search_smoothing(
    log_iws: chex.Array,
    target_ess: float = 0.0,
    tol=1e-3,
    max_steps=1000,
) -> chex.Array:
    tempering = lambda x, temp: x / temp
    batch_size = log_iws.shape[0]  # type: ignore

    # Check if tempering is needed
    if done := ess(log_iws=log_iws) / batch_size >= target_ess:
        return log_iws

    # Search for a suitable range of search_min and search_max
    search_min = 1.0
    search_max = 10.0
    while ess(tempering(log_iws, search_max)) / batch_size < target_ess:
        search_min *= 10.0
        search_max *= 10.0

    new_log_iws = jnp.copy(log_iws)
    final_temp = 1.0
    steps = 0
    while not done:
        steps += 1
        mid = (search_min + search_max) / 2

        new_log_iws = tempering(log_iws, mid)  # (bs,)
        new_ess = ess(log_iws=new_log_iws) / batch_size
        done = jax.lax.abs(new_ess - target_ess) < tol
        if done:
            final_temp = mid

        if new_ess > target_ess:
            search_max = mid
        else:
            search_min = mid

        if steps > max_steps:
            print(f"Warning: Binary search failed in {max_steps} steps")
            break

    return new_log_iws, final_temp


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
