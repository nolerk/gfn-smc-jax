import os
from typing import List

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.random as random
import numpy as np
from functools import partial
import distrax
from matplotlib import pyplot as plt
from scipy.stats import wishart

from targets.base_target import Target
from utils.path_utils import project_path
from algorithms.common.bounded import rejection_sample_domain


class Cube(Target):
    def __init__(
        self,
        dim,
        log_prob_fn=None,
        min_coord=0.0,
        max_coord=1.0,
        mask_outside=True,  # whether to set p(x)=0 outside the domain
        sample_fn=None,
        sample_bounds=None,
    ) -> None:
        self._dim = dim
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.log_prob_fn = log_prob_fn
        self.mask_outside = mask_outside
        self.sample_fn = sample_fn

        log_Z = self._compute_true_logZ()
        super().__init__(dim, log_Z, can_sample=sample_fn is not None)

    def is_inside(self, x: chex.Array) -> bool:
        return jnp.all((x >= self.min_coord) & (x <= self.max_coord), axis=-1)

    def projection_fn(self, x: chex.Array) -> chex.Array:
        return jnp.clip(x, min=self.min_coord, max=self.max_coord)

    def rd_to_cube(self, x: chex.Array):
        return (self.max_coord - self.min_coord) * jax.nn.sigmoid(x) + self.min_coord

    def cube_to_rd(self, x: chex.Array):
        x = (x - self.min_coord) / (self.max_coord - self.min_coord)
        # eps = 1e-6
        # x = jnp.clip(x, min=eps, max=1 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    def sample(
        self, seed: chex.PRNGKey, sample_shape: chex.Shape, constrained: bool = False
    ) -> chex.Array:
        # constrained = True - defined in [a,b]^d
        # constrained = False - defined in R^d via change of variables
        if not constrained:
            samples = self.sample(seed, sample_shape, constrained=True)
            return self.cube_to_rd(samples)

        return rejection_sample_domain(
            seed, sample_shape, self.sample_fn, self.is_inside, self.dim
        )

    def log_prob(self, x: chex.Array, constrained: bool = False) -> chex.Array:
        # constrained = True - defined in [a,b]^d
        # constrained = False - defined in R^d via change of variables
        if not constrained:
            log_jacobian = jnp.sum(
                jax.nn.log_sigmoid(x) + jax.nn.log_sigmoid(-x), axis=-1
            ) + x.shape[-1] * jnp.log(self.max_coord - self.min_coord)
            return self.log_prob(self.rd_to_cube(x), constrained=True) + log_jacobian

        batched = x.ndim == 2
        if not batched:
            x = x[None]

        log_prob = self.log_prob_fn(x)
        if self.mask_outside:
            log_prob = jnp.where(self.is_inside(x), log_prob, -jnp.inf)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def _compute_true_logZ(
        self, num_samples: int = 100000, seed: chex.PRNGKey = None
    ) -> float:
        if seed is None:
            seed = jax.random.PRNGKey(42)
        shape = (num_samples, self.dim)
        seed, subkey = random.split(seed)
        samples = jax.random.uniform(
            subkey, shape=shape, minval=self.min_coord, maxval=self.max_coord
        )
        log_probs = self.log_prob(samples, constrained=True)
        mask = jnp.isfinite(log_probs)
        valid_log_probs = log_probs[mask]

        log_volume = self.dim * jnp.log(float(self.max_coord - self.min_coord))
        log_integral = (
            logsumexp(valid_log_probs) - jnp.log(valid_log_probs.shape[0]) + log_volume
        )
        return float(log_integral)

    def visualise(
        self,
        samples: chex.Array = None,
        axes: List[plt.Axes] = None,
        show=False,
        clip=False,
        constrained=True,
        map_samples=True,
    ) -> None:
        # samples are mapped to cube if constrained=True and map_samples=True
        plt.close()
        if constrained:
            boarder = [self.min_coord - 1, self.max_coord + 1]
        else:
            boarder = [
                self.cube_to_rd(jnp.array(self.min_coord + 1e-3)),
                self.cube_to_rd(jnp.array(self.max_coord - 1e-3)),
            ]
        log_prob_fn = (
            partial(self.log_prob, constrained=True) if constrained else self.log_prob
        )
        if self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()

            x, y = jnp.meshgrid(
                jnp.linspace(boarder[0], boarder[1], 100),
                jnp.linspace(boarder[0], boarder[1], 100),
            )
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(log_prob_fn(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            ax.contourf(x, y, pdf_values, levels=20, cmap="viridis")
            if samples is not None:
                if constrained and map_samples:
                    samples = self.rd_to_cube(samples)
                plt.scatter(
                    samples[:300, 0], samples[:300, 1], c="r", alpha=0.5, marker="x"
                )
            if constrained:
                # Draw a rectangle (box) corresponding to the support [min_coord, max_coord] for both axes
                import matplotlib.patches as patches

                rect = patches.Rectangle(
                    (self.min_coord, self.min_coord),
                    self.max_coord - self.min_coord,
                    self.max_coord - self.min_coord,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="none",
                )
                ax.add_patch(rect)
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig(os.path.join(project_path('./figures/'), f"gmm2D.pdf"), bbox_inches='tight', pad_inches=0.1)
            wb = {"figures/vis": [fig]}
            if show:
                plt.show()

            return wb

        else:

            return {}
