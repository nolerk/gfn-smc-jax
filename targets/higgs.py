import chex
import jax
import jax.numpy as jnp

from targets.cube import Cube


class Higgs(Cube):
    def __init__(self) -> None:
        super().__init__(
            dim=3,
            log_prob_fn=self._log_box,
            min_coord=0.0,
            max_coord=1.0,
            sample_fn=lambda seed, sample_shape: jnp.zeros(
                (*sample_shape, 3)
            ),  # we don't need sampling here
        )

    def _f_box(self, x: chex.Array, s12, s23, s1, s2, s3, s4, m1_2, m2_2, m3_2, m4_2):
        t1 = x[..., 0]
        t2 = x[..., 1]
        t3 = x[..., 2]
        return (
            -s12 * t2
            - s23 * t1 * t3
            - s1 * t1
            - s2 * t1 * t2
            - s3 * t2 * t3
            - s4 * t3
            + (1 + t1 + t2 + t3) * (t1 * m1_2 + t2 * m2_2 + t3 * m3_2 + m4_2)
        )

    def _log_box(self, x: chex.Array):
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 125**2
        s12 = 130**2
        s23 = -(130**2)
        mt_2 = 175**2

        int_x = (
            1 / self._f_box(x, s12, s23, s1, s2, s3, s4, mt_2, mt_2, mt_2, mt_2) ** 2
            + 1 / self._f_box(x, s23, s12, s2, s3, s4, s1, mt_2, mt_2, mt_2, mt_2) ** 2
            + 1 / self._f_box(x, s12, s23, s3, s4, s1, s2, mt_2, mt_2, mt_2, mt_2) ** 2
            + 1 / self._f_box(x, s23, s12, s4, s1, s2, s3, mt_2, mt_2, mt_2, mt_2) ** 2
        )
        return jnp.log(int_x)

    def _compute_true_logZ(self) -> float:
        mant = 1.9369640238
        return jnp.log(mant) - 10 * jnp.log(10)


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    higgs = Higgs()
    print(jnp.exp(higgs.log_Z))
    higgs.visualise(show=True, constrained=True, map_samples=False)
