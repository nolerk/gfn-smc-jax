"""
Annealed Importance Sampling (AIS)
For further details see: Neal, R. M. (1998). Annealed importance sampling.
Statistics and Computing, 11(2), 125-139.
https://arxiv.org/abs/physics/9803008

Code builds on https://github.com/google-deepmind/annealed_flow_transport
"""

import time
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

import algorithms.common.types as tp
from algorithms.common import flow_transport, markov_kernel
from algorithms.common.eval_methods.sis_methods import get_eval_fn
from eval.utils import extract_last_entry
from targets.base_target import Target
from utils.logger import log
from utils.print_utils import print_results

Array = tp.Array
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
assert_equal_shape = chex.assert_equal_shape
assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes
AlgoResultsTuple = tp.AlgoResultsTuple


def inner_loop_ais(
    key: RandomKey,
    markov_kernel_apply: MarkovKernelApply,
    samples: Array,
    log_weights: Array,
    log_density: LogDensityByStep,
    step: int,
) -> Tuple[Array, Array, Array, Array]:
    """Inner loop of AIS - single annealing step.
    
    AIS differs from SMC in that there is NO resampling.
    Each particle maintains its own independent weight trajectory.

    Args:
      key: A JAX random key.
      markov_kernel_apply: functional that applies the Markov transition kernel.
      samples: Array containing samples.
      log_weights: Array containing log_weights.
      log_density: function returning the log_density of a sample at given step.
      step: int giving current step of algorithm.

    Returns:
      samples_final: samples after MCMC transition.
      log_weights_final: updated log_weights.
      log_normalizer_increment: Scalar log of normalizing constant increment.
      acceptance_tuple: Acceptance rates of samplers.
    """
    # Compute weight update: w_t = w_{t-1} * p_{t-1}(x) / p_t(x)
    # In log space: log_w_t = log_w_{t-1} + log p_{t-1}(x) - log p_t(x)
    log_density_current = log_density(step, samples)
    log_density_previous = log_density(step - 1, samples)
    deltas = log_density_previous - log_density_current
    assert_equal_shape([log_density_current, log_density_previous, deltas])
    
    # Update weights (no resampling in vanilla AIS)
    log_weights_new = log_weights + deltas
    
    # Compute normalizer increment for this step
    # Using self-normalized importance sampling estimator
    log_normalizer_increment = jax.scipy.special.logsumexp(deltas) - jnp.log(deltas.shape[0])
    elbo_increment = jnp.mean(-deltas)
    
    # MCMC transition at current temperature
    markov_samples, acceptance_tuple = markov_kernel_apply(step, key, samples)

    return markov_samples, log_weights_new, (log_normalizer_increment, elbo_increment), acceptance_tuple


def get_short_inner_loop_ais(
    markov_kernel_by_step: MarkovKernelApply, 
    density_by_step: LogDensityByStep
):
    """Get a short version of inner loop for JIT compilation."""

    def short_inner_loop(
        rng_key: RandomKey, 
        loc_samples: Array, 
        loc_log_weights: Array, 
        loc_step: int
    ):
        return inner_loop_ais(
            rng_key,
            markov_kernel_by_step,
            loc_samples,
            loc_log_weights,
            density_by_step,
            loc_step,
        )

    return short_inner_loop


def outer_loop_ais(
    density_by_step: LogDensityByStep,
    initial_sampler: InitialSampler,
    target: Target,
    markov_kernel_by_step: MarkovKernelApply,
    key: RandomKey,
    cfg,
) -> AlgoResultsTuple:
    """The outer loop for Annealed Importance Sampling.

    AIS runs multiple independent chains from a proposal distribution,
    transitioning through a sequence of annealed distributions using MCMC,
    and accumulating importance weights without resampling.

    Args:
      density_by_step: The log density for each annealing step.
      initial_sampler: A function that produces the initial samples.
      target: The target distribution.
      markov_kernel_by_step: Markov transition kernel for each annealing step.
      key: A JAX random key.
      cfg: A ConfigDict containing the configuration.

    Returns:
      An AlgoResultsTuple containing a summary of the results.
    """
    key, subkey = jax.random.split(key)
    alg_cfg = cfg.algorithm
    num_temps = alg_cfg.num_temps
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Initialize samples from the proposal distribution
    samples = initial_sampler(seed=jax.random.PRNGKey(0), sample_shape=(alg_cfg.batch_size,))
    
    # Initialize uniform weights: log(1/N) for each particle
    log_weights = -jnp.log(alg_cfg.batch_size) * jnp.ones(alg_cfg.batch_size)

    # JIT compile the inner loop
    inner_loop_jit = jax.jit(get_short_inner_loop_ais(markov_kernel_by_step, density_by_step))

    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    ln_z = 0.0
    elbo = 0.0
    start_time = time.time()

    acceptance_hmc = []
    acceptance_rwm = []

    # Main AIS loop - iterate through annealing schedule
    for step in range(1, num_temps):
        subkey, key = jax.random.split(key)
        samples, log_weights, ln_z_inc, acceptance = inner_loop_jit(
            subkey, samples, log_weights, step
        )
        ln_z_inc, elbo_inc = ln_z_inc
        acceptance_hmc.append(float(np.asarray(acceptance[0])))
        acceptance_rwm.append(float(np.asarray(acceptance[1])))
        ln_z += ln_z_inc
        elbo += elbo_inc

    finish_time = time.time()
    delta_time = finish_time - start_time

    # Compute final estimates
    # Self-normalized importance sampling estimate of log Z
    ln_z_final = jax.scipy.special.logsumexp(log_weights) - jnp.log(alg_cfg.batch_size)
    
    # Effective sample size
    ess = jnp.exp(2 * jax.scipy.special.logsumexp(log_weights) - 
                  jax.scipy.special.logsumexp(2 * log_weights))
    ess_normalized = ess / alg_cfg.batch_size

    # Number of function evaluations
    nfe = 2 * alg_cfg.batch_size * (alg_cfg.num_temps - 1)

    logger = eval_fn(samples, elbo, ln_z_final, None, None)
    
    logger["stats/wallclock"] = [delta_time]
    logger["stats/nfe"] = [nfe]
    logger["ESS/forward"] = [ess_normalized]
    logger["other/avg_acceptance_hmc"] = [sum(acceptance_hmc) / len(acceptance_hmc)]
    logger["other/avg_acceptance_rwm"] = [sum(acceptance_rwm) / len(acceptance_rwm)]

    print_results(0, logger, cfg)

    if cfg.use_logger:
        log(extract_last_entry(logger))

    return AlgoResultsTuple(
        test_samples=samples,
        test_log_weights=log_weights,
        ln_Z_estimate=ln_z_final,
        ELBO_estimate=elbo,
        MMD=logger["discrepancies/mmd"][-1] if "discrepancies/mmd" in logger else None,
        delta_time=delta_time,
        initial_time_diff=0.0,
    )