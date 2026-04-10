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


class Higgs(Target):
    def __init__(
        self,
        dim=3,
        num_components=1,
        min_coord=0.0,
        max_coord=1.0,
        can_sample=False,
    ) -> None:
        self._dim = dim
        self.min_coord = min_coord
        self.max_coord = max_coord

        self.mixture_distribution = distrax.Uniform(
            low=jnp.full(dim, self.min_coord),
            high=jnp.full(dim, self.max_coord)
        )

        log_Z = self._compute_true_logZ()
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
    
    def _f_box(self, x: chex.Array, s12, s23, s1, s2, s3, s4, m1_2, m2_2, m3_2, m4_2):
        t1 = x[..., 0]
        t2 = x[..., 1]
        t3 = x[..., 2]
        return (
            -s12 * t2 - s23 * t1 * t3 - s1 * t1 - s2 * t1 * t2 - s3 * t2 * t3
            - s4 * t3 + (1 + t1 + t2 + t3) * (t1 * m1_2 + t2 * m2_2 + t3 * m3_2 + m4_2)
        )
    
    def _box(self, x: chex.Array):
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 125 ** 2
        s12 = 130 ** 2
        s23 = -130**2
        mt_2 = 175 ** 2

        return (
            1 / self._f_box(x, s12, s23, s1, s2, s3, s4, mt_2, mt_2, mt_2, mt_2) ** 2
            + 1 / self._f_box(x, s23, s12, s2, s3, s4, s1, mt_2, mt_2, mt_2, mt_2) ** 2
            + 1 / self._f_box(x, s12, s23, s3, s4, s1, s2, mt_2, mt_2, mt_2, mt_2) ** 2
            + 1 / self._f_box(x, s23, s12, s4, s1, s2, s3, mt_2, mt_2, mt_2, mt_2) ** 2
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

        log_prob = jnp.log(self._box(x))

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def _compute_true_logZ(
        self
    ) -> float:
        """
        Estimate the normalization constant Z (partition function) of the constrained density.
        
        Returns:
            float: Estimated log Z = log ∫ p(x) dx over [min_coord, max_coord]^d
        """
        mant = 1.9369640238
        return jnp.log(mant) - 10 * jnp.log(10)
    
    def sample(
        self, seed: chex.PRNGKey, sample_shape: chex.Shape, constrained: bool = False
    ) -> chex.Array:
        # constrained = True - defined in [a,b]^d
        # constrained = False - defined in R^d via change of variables
        if not constrained:
            samples = self.sample(seed, sample_shape, constrained=True)
            return self.cube_to_rd(samples)

        return rejection_sample_domain(
            seed, sample_shape, self.mixture_distribution, self.is_inside, self.dim
        )
    
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

if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    cube = Higgs(dim=3, min_coord=0.0, max_coord=1.0)
    print(jnp.exp(cube.log_Z))
