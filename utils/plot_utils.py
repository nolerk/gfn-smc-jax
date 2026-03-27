"""Code builds on https://github.com/lollcat/fab-jax"""

import itertools
from typing import Optional, Tuple

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_contours_2D(
    log_prob_func,
    dim: int,
    ax: plt.Axes,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: tuple[float, float] = (-3, 3),
    levels: int = 20,
    n_points: int = 200,
    log: bool = False,
):
    """Plot the contours of a 2D log prob function."""
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
    if log:
        z = log_probs
    else:
        z = jnp.exp(log_probs)
    z = z.reshape(n_points, n_points)
    ax.contourf(x1, x2, z, levels=levels)


def plot_marginal_pair(
    samples: chex.Array,
    ax: plt.Axes,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5, 5),
    alpha: float = 0.5,
    max_points: int = 500,
    color: str = "r",
    marker: str = "x",
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.scatter(
        samples[:max_points, marginal_dims[0]],
        samples[:max_points, marginal_dims[1]],
        color=color,
        alpha=alpha,
        marker=marker,
    )


def visualise_trajectories(
    trajectories,
    target,
    dims=(0, 1),
    device="cpu",
    alpha=0.8,
    prefix="",
    show=False,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    samples = trajectories[:, -1]
    samples = samples[:, dims]
    min_bounds = samples.min(axis=0) - 1.5
    max_bounds = samples.max(axis=0) + 1.5

    batch_size = trajectories.shape[0]

    cmap = plt.get_cmap("viridis")
    colors = cmap(jnp.linspace(0, 1, batch_size))

    marker_size = 50

    x = jnp.linspace(min_bounds[0], max_bounds[0], 50)
    y = jnp.linspace(min_bounds[1], max_bounds[1], 50)

    if trajectories.shape[-1] == 2:
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
        grid = jax.device_put(grid, device)
        pdf = jnp.exp(target.log_prob(grid)).reshape(X.shape)
        pdf = jax.device_put(pdf, jax.devices("cpu")[0])
        levels = jnp.linspace(pdf.min(), pdf.max(), 20)
        ax.contourf(X, Y, pdf, levels=levels, cmap="viridis", alpha=0.5)

    # 3. Loop through and plot each trajectory
    for i in range(batch_size):
        color = colors[i]

        # 4. Slice the valid part of the trajectory
        valid_traj = trajectories[i]

        # 5. Plot the trajectory line
        ax.plot(
            valid_traj[:, dims[0]],
            valid_traj[:, dims[1]],
            color=color,
            alpha=alpha,
        )

        # 6. Mark the start and end points
        ax.scatter(
            valid_traj[0, dims[0]],
            valid_traj[0, dims[1]],
            color=color,
            marker="o",
            s=marker_size,
            edgecolors="black",
            zorder=3,
        )
        ax.scatter(
            valid_traj[-1, dims[0]],
            valid_traj[-1, dims[1]],
            color=color,
            marker="X",
            s=marker_size,
            edgecolors="black",
            zorder=3,
        )

    ax.set_xlabel(f"x{dims[0]+1}")
    ax.set_ylabel(f"x{dims[1]+1}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    ax.set_aspect("equal", adjustable="box")

    wb = {f"figures/{prefix + '_' if prefix else ''}vis": [fig]}
    if show:
        plt.show()
    else:
        plt.close()
    return wb
