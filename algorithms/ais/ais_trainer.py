"""
Annealed Importance Sampling (AIS) Trainer
For further details see: Neal, R. M. (1998). Annealed importance sampling.
Statistics and Computing, 11(2), 125-139.
https://arxiv.org/abs/physics/9803008
"""

import distrax
import jax
import jax.numpy as jnp

from algorithms.common import flow_transport, markov_kernel
from algorithms.ais import ais


def ais_trainer(cfg, target):
    """Trainer for Annealed Importance Sampling.
    
    Args:
        cfg: Configuration dictionary.
        target: Target distribution.
    """
    final_log_density = target.log_prob
    dim = target.dim
    alg_cfg = cfg.algorithm
    mcmc_cfg = cfg.algorithm.mcmc

    # Define the initial/proposal distribution (typically Gaussian)
    initial_density = distrax.MultivariateNormalDiag(
        jnp.ones(dim) * alg_cfg.init_mean, 
        jnp.ones(dim) * alg_cfg.init_std
    )

    log_density_initial = initial_density.log_prob
    initial_sampler = initial_density.sample

    num_temps = alg_cfg.num_temps
    
    # Set up the geometric annealing schedule
    density_by_step = flow_transport.GeometricAnnealingSchedule(
        log_density_initial, final_log_density, num_temps
    )
    
    # Set up the MCMC transition kernel
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
        mcmc_cfg, density_by_step, num_temps
    )

    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key)

    ais.outer_loop_ais(
        density_by_step=density_by_step,
        initial_sampler=initial_sampler,
        markov_kernel_by_step=markov_kernel_by_step,
        key=key,
        target=target,
        cfg=cfg,
    )