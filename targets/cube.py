import os
from typing import List

import chex
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from functools import partial
import distrax
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wishart

from targets.base_target import Target
from utils.path_utils import project_path


class Cube(Target):
    def __init__(
        self,
        dim,
        num_components,
        min_coord=0.0,
        max_coord=1.0,
        scale=1.0,
        can_sample=True,
        sample_bounds=None,
    ) -> None:
        self._dim = dim
        self.min_coord = min_coord
        self.max_coord = max_coord

        seed = jax.random.PRNGKey(1)
        degree_of_freedom_wishart = dim + 2

        seed, subkey = random.split(seed)
        # set mixture components
        self.locs = jax.random.uniform(
            subkey,
            minval=min_coord,
            maxval=max_coord,
            shape=(num_components, dim),
        )
        self.covariances = []
        for _ in range(num_components):
            seed, subkey = random.split(seed)

            # Set the random seed for Scipy
            seed_value = random.randint(key=subkey, shape=(), minval=0, maxval=2**30)
            np.random.seed(seed_value)

            cov_matrix = wishart.rvs(
                df=degree_of_freedom_wishart, scale=scale * jnp.eye(dim)
            )
            self.covariances.append(cov_matrix)
        self.covariances = jnp.array(self.covariances)

        component_dist = distrax.MultivariateNormalFullCovariance(
            self.locs, self.covariances
        )
        mixture_weights = distrax.Categorical(
            logits=jnp.ones(num_components) / num_components
        )
        self.mixture_distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_weights,
            components_distribution=component_dist,
        )

        log_Z = self.compute_true_logZ()
        super().__init__(dim, log_Z, can_sample)

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
        # constrained = True meaning it is defined in [a,b]^d
        if constrained:
            num_samples = int(np.prod(sample_shape))
            samples = []
            num_accepted = 0
            while num_accepted < num_samples:
                seed, subkey = random.split(seed)
                y = self.mixture_distribution.sample(
                    seed=subkey, sample_shape=(num_samples,)
                )
                mask = self.is_inside(y)
                accepted = y[mask]
                samples.append(accepted)
                num_accepted += int(np.prod(accepted.shape[:-1]))
            samples = jnp.concatenate(samples, axis=0)
            return samples[:num_samples].reshape(sample_shape + (self.dim,))
        return self.cube_to_rd(self.sample(seed, sample_shape, constrained=True))

    def compute_true_logZ(
        self, num_samples: int = 100000, seed: chex.PRNGKey = None
    ) -> float:
        """
        Estimate the normalization constant Z (partition function) of the constrained density
        using Monte Carlo integration.

        Args:
            num_samples (int): Number of Monte Carlo samples to use.
            seed: (chex.PRNGKey or None): Optional random seed.

        Returns:
            float: Estimated log Z = log ∫ p(x) dx over [min_coord, max_coord]^d
        """
        if seed is None:
            seed = jax.random.PRNGKey(42)
        # Uniformly sample points in the cube
        shape = (num_samples, self.dim)
        seed, subkey = random.split(seed)
        samples = jax.random.uniform(
            subkey, shape=shape, minval=self.min_coord, maxval=self.max_coord
        )
        # Compute probability under the constrained mixture model
        log_probs = self.log_prob(samples, constrained=True)
        # Since log_prob can return -inf for out-of-domain, mask them out
        mask = jnp.isfinite(log_probs)
        valid_log_probs = log_probs[mask]
        # Use logsumexp for numerical stability
        from scipy.special import logsumexp

        log_volume = self.dim * np.log(float(self.max_coord - self.min_coord))
        # Estimate integral
        log_integral = (
            logsumexp(np.array(valid_log_probs))
            - np.log(valid_log_probs.shape[0])
            + log_volume
        )
        return float(log_integral)

    def log_prob(self, x: chex.Array, constrained: bool = False) -> chex.Array:
        # constrained = True meaning it is defined in [a,b]^d
        if constrained:
            batched = x.ndim == 2
            if not batched:
                x = x[None]

            log_prob = self.mixture_distribution.log_prob(x)
            log_prob = jnp.where(self.is_inside(x), log_prob, -jnp.inf)

            if not batched:
                log_prob = jnp.squeeze(log_prob, axis=0)
            return log_prob
        log_jacobian = jnp.sum(
            jax.nn.log_sigmoid(x) + jax.nn.log_sigmoid(-x), axis=-1
        ) + x.shape[-1] * jnp.log(self.max_coord - self.min_coord)
        return self.log_prob(self.rd_to_cube(x), constrained=True) + log_jacobian

    def visualise(
        self,
        samples: chex.Array = None,
        axes: List[plt.Axes] = None,
        show=False,
        clip=False,
        constrained=True,
    ) -> None:
        # passed samples by default are from R^d to not change visualization code for baselines
        plt.close()
        boarder = [self.min_coord - 1, self.max_coord + 1]
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
                if constrained:
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
                    linewidth=2,
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


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    cube = Cube(dim=2, num_components=10, min_coord=0.0, max_coord=1.0, scale=0.001)
    print(jnp.exp(cube.log_Z))
    samples = cube.sample(key, (100,))
    cube.visualise(samples, show=True, constrained=True)
