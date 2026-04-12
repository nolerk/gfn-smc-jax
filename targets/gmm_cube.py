import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import distrax
from scipy.stats import wishart
from targets.cube import Cube


class GMMCube(Cube):
    def __init__(
        self,
        dim,
        num_components,
        min_coord=0.0,
        max_coord=1.0,
        mask_outside=True,
        scale=1.0,
        sample_bounds=None,
    ) -> None:
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
        dist = distrax.MixtureSameFamily(
            mixture_distribution=mixture_weights,
            components_distribution=component_dist,
        )
        super().__init__(
            dim,
            log_prob_fn=dist.log_prob,
            min_coord=min_coord,
            max_coord=max_coord,
            mask_outside=mask_outside,
            sample_fn=dist.sample,
            sample_bounds=sample_bounds,
        )


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    cube = GMMCube(dim=2, num_components=10, min_coord=0.0, max_coord=1.0, scale=0.001)
    print(jnp.exp(cube.log_Z))
    # samples = cube.sample(key, (300,), constrained=True)

    @jax.jit
    def sample():
        return cube.sample(key, (300,), constrained=True)

    samples = sample()
    cube.visualise(samples, show=True, constrained=True, map_samples=False)
