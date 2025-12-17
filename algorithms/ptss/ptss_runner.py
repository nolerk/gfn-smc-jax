import jax
import jax.numpy as jnp

from algorithms.ptss.ptss import parallel_tempering_with_slice_sampling


def ptss_sampling_runner(cfg, target):

    key = jax.random.PRNGKey(cfg.seed)
    x_start = jax.random.normal(key, (cfg.batch_size, target.dim)) * cfg.init_scale

    temps = jnp.array(cfg.temperatures)

    chain = parallel_tempering_with_slice_sampling(
        x_start=x_start,
        log_density=target.log_prob,
        temperatures=temps,
        chain_length=cfg.chain_length,
        swap_freq=cfg.swap_freq,
        alpha=cfg.alpha,
        seed=cfg.seed,
        propose_region_limits=cfg.propose_region_limits,
    )

    return chain

