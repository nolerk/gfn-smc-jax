"""Code builds on https://github.com/lollcat/fab-jax"""

import itertools
from typing import Optional, Tuple

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_contours_2D(
    log_prob_func,
    dim: int,
    ax: Optional[plt.Axes] = None,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: tuple[float, float] = (-3, 3),
    levels: int = 20,
    n_points: int = 200,
):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = np.linspace(bounds[0], bounds[1], n_points)
    x_points_dim2 = np.linspace(bounds[0], bounds[1], n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))

    def sliced_log_prob(x: chex.Array):
        _x = jnp.zeros((x.shape[0], dim))
        _x = _x.at[:, marginal_dims].set(x)
        return log_prob_func(_x)

    log_probs = sliced_log_prob(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels)


def plot_marginal_pair(
    samples: chex.Array,
    ax: Optional[plt.Axes] = None,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5, 5),
    alpha: float = 0.5,
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)
