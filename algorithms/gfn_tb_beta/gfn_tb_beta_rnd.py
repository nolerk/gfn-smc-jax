from typing import Callable, Literal

import distrax
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
from flax.training.train_state import TrainState
from functools import partial

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.gfn_tb_beta.sampling_utils import binary_search_smoothing


def sample_kernel_fn(key_gen, alpha, beta, a, b):
    key, key_gen = jax.random.split(key_gen)
    s = a + (b - a) * jax.random.beta(key, alpha, beta, shape=(alpha.shape[0],))
    return s, key_gen


def log_prob_kernel_fn(x, alpha, beta, a, b):
    x = (x - a) / (b - a)
    dist = npdist.Independent(npdist.Beta(alpha, beta), 1)
    return dist.log_prob(x) - alpha.shape[0] * jnp.log(b - a)


def per_sample_rnd(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    prior_to_target=True,
):
    (dim,) = aux_tuple
    dt = 1.0 / num_steps

    sample_kernel = partial(sample_kernel_fn, a=target.min_coord, b=target.max_coord)
    log_prob_kernel = partial(
        log_prob_kernel_fn, a=target.min_coord, b=target.max_coord
    )

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s = jax.lax.stop_gradient(s)

        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
        else:
            langevin = jnp.zeros(dim)
        model_output, _ = model_state.apply_fn(
            params, s, t * jnp.ones(1), langevin, is_fwd=True
        )

        fwd_alpha, fwd_beta = jnp.split(model_output, [dim], axis=-1)
        s_next, key_gen = sample_kernel(key_gen, fwd_alpha, fwd_beta)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_alpha, fwd_beta)

        model_output_bwd, _ = model_state.apply_fn(
            params,
            s_next,
            t_next * jnp.ones(1),
            is_fwd=False,
        )
        bwd_alpha, bwd_beta = jnp.split(model_output_bwd, [dim], axis=-1)
        bwd_log_prob = log_prob_kernel(s, bwd_alpha, bwd_beta)

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s_next = jax.lax.stop_gradient(s_next)
        model_output_bwd, _ = model_state.apply_fn(
            params,
            s_next,
            t_next * jnp.ones(1),
            is_fwd=False,
        )
        bwd_alpha, bwd_beta = jnp.split(model_output_bwd, [dim], axis=-1)
        s, key_gen = sample_kernel(key_gen, bwd_alpha, bwd_beta)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = log_prob_kernel(s, bwd_alpha, bwd_beta)

        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
        else:
            langevin = jnp.zeros(dim)
        model_output, _ = model_state.apply_fn(
            params, s, t * jnp.ones(1), langevin, is_fwd=True
        )

        fwd_alpha, fwd_beta = jnp.split(model_output, [dim], axis=-1)
        fwd_log_prob = log_prob_kernel(s_next, fwd_alpha, fwd_beta)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob)
        return next_state, per_step_output

    if prior_to_target:
        init_x = input_state
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(num_steps)
        )
        terminal_x, _ = aux
    else:
        terminal_x = input_state
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )

    trajectory, fwd_log_prob, bwd_log_prob = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob


def rnd(
    key_gen,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    prior_to_target=True,
    initial_dist: distrax.Distribution | None = None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    if prior_to_target:
        if initial_dist is None:
            input_states = jnp.zeros((batch_size, target.dim))
        else:
            key, key_gen = jax.random.split(key_gen)
            input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)

    terminal_xs, trajectories, fwd_log_probs, bwd_log_probs = jax.vmap(
        per_sample_rnd,
        in_axes=(0, None, None, 0, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps,
        use_lp,
        prior_to_target,
    )

    if not prior_to_target:
        trajectories = trajectories[:, ::-1]
        fwd_log_probs = fwd_log_probs[:, ::-1]
        bwd_log_probs = bwd_log_probs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)
    if initial_dist is None:
        init_fwd_log_probs = jnp.zeros(batch_size)
    else:
        init_fwd_log_probs = initial_dist.log_prob(trajectories[:, 0])

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories[:, -1],
        log_pfs_over_pbs.sum(1) + init_fwd_log_probs,
        jnp.zeros_like(log_rewards),
        -log_rewards,
    )


def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[
        [RandomKey, TrainState, ModelParams], tuple[Array, Array, Array, Array]
    ],
    loss_type: Literal["tb", "lv"],
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
    importance_weighting: bool = False,
    target_ess: float = 0.0,
):
    terminal_xs, running_costs, _, terminal_costs = rnd_partial(
        key, model_state, params
    )
    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    log_ratio = running_costs - log_rewards * invtemp  # log_pfs - log_pbs - log_rewards
    if loss_type == "tb":
        losses = log_ratio + params["params"]["logZ"]
    else:  # loss_type == "lv"
        losses = log_ratio - jnp.mean(log_ratio)

    if huber_delta is not None:
        losses = jnp.where(
            jnp.abs(losses) <= huber_delta,
            jnp.square(losses),
            huber_delta * jnp.abs(losses) - 0.5 * huber_delta**2,
        )
    else:
        losses = jnp.square(losses)

    if importance_weighting:
        smoothed_log_iws, _ = binary_search_smoothing(-log_ratio, target_ess)
        normalized_iws = jax.nn.softmax(smoothed_log_iws, axis=-1)
        loss = jnp.sum(losses * normalized_iws)
    else:
        loss = jnp.mean(losses)

    return loss, (
        terminal_xs,
        jax.lax.stop_gradient(-running_costs),  # log_pbs - log_pfs
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(losses),  # losses
    )


if __name__ == "__main__":
    d = 2
    s = sample_kernel(jax.random.PRNGKey(0), 0.5 * jnp.ones((d,)), 0.5 * jnp.ones((d,)))
    print(s.shape)
