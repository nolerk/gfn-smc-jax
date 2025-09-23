import warnings
from functools import partial
from typing import Callable, Literal

import chex
import jax
import jax.numpy as jnp
from jax import random

from algorithms.gfn_tb.sampling_utils import ess


@jax.jit
def binary_search_smoothing_jit(
    log_iws: chex.Array,
    target_ess: float = 0.0,
    tol=1e-3,
    max_steps=1000,
) -> chex.Array:
    batch_size = log_iws.shape[0]  # type: ignore

    # Helper closures (JAX-friendly)
    def normalize_ess(_log_iws: chex.Array) -> chex.Array:
        return ess(log_iws=_log_iws) / batch_size

    tol_f = jnp.asarray(tol, dtype=log_iws.dtype)

    # Early exit if already meets target ESS
    init_norm_ess = normalize_ess(log_iws)

    def _early_return(_: None):
        return log_iws, jnp.asarray(1.0, dtype=log_iws.dtype)

    def _continue(_: None):
        # Expand search range so that ESS(log_iws / search_max) >= target_ess
        init_search_min = jnp.asarray(1.0, dtype=log_iws.dtype)
        init_search_max = jnp.asarray(10.0, dtype=log_iws.dtype)

        def range_cond_fun(state):
            _, search_max, _ = state
            cur_ess = normalize_ess(log_iws / search_max)
            return cur_ess < target_ess

        def range_body_fun(state):
            search_min, search_max, steps = state
            return search_min * 10.0, search_max * 10.0, steps + 1

        search_min, search_max, _ = jax.lax.while_loop(
            range_cond_fun,
            range_body_fun,
            (init_search_min, init_search_max, jnp.asarray(0, dtype=jnp.int32)),
        )

        # Binary search for temperature achieving ESS close to target
        init_state = (
            search_min,  # search_min
            search_max,  # search_max
            jnp.asarray(0, dtype=jnp.int32),  # steps
            jnp.asarray(False),  # done
            jnp.asarray(1.0, dtype=log_iws.dtype),  # final_temp
        )

        def bin_cond_fun(state):
            _, _, step, done, _ = state
            return (~done) & (step < max_steps)

        def bin_body_fun(state):
            search_min, search_max, step, _, _ = state
            mid = (search_min + search_max) / 2.0

            tempered_log_iws = log_iws / mid
            new_ess = normalize_ess(tempered_log_iws)
            is_converged = jnp.abs(new_ess - target_ess) < tol_f

            # Update the bracket based on whether ESS is above or below target
            go_left = new_ess > target_ess
            new_search_max = jnp.where(go_left, mid, search_max)
            new_search_min = jnp.where(go_left, search_min, mid)

            return new_search_min, new_search_max, step + 1, is_converged, mid

        search_min, search_max, step, done, final_temp = jax.lax.while_loop(
            bin_cond_fun, bin_body_fun, init_state
        )

        # print warning if not converged
        def _print_warning(_: None):
            warnings.warn(f"Binary search failed in {max_steps} steps")
            return None

        jax.lax.cond(step >= max_steps, _print_warning, lambda _: None, operand=None)

        # Choose final temperature: best match if found, else mid of bracket
        return log_iws / final_temp, final_temp

    return jax.lax.cond(init_norm_ess >= target_ess, _early_return, _continue, operand=None)
