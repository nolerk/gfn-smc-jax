from typing import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams


def sample_kernel(key_gen, mean, scale):
    key, key_gen = jax.random.split(key_gen)
    eps = jax.random.normal(key, shape=(mean.shape[0],))
    return mean + scale * eps, key_gen


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def get_partial_energy(s, t, sigma_t, log_prob):
    ref_log_var = jnp.log((sigma_t**2) * jnp.array(t))
    log_p_ref = -0.5 * (
        jnp.log(2 * jnp.pi) + ref_log_var + jnp.exp(-ref_log_var) * jnp.square(s)
    ).sum(-1)
    return (1 - t) * log_p_ref + t * log_prob


def per_sample_rnd_pinned_brownian(
    seed,
    model_state,
    params,
    sde_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    dim, _ = sde_tuple

    sigmas = noise_schedule
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        sigma_t = sigmas(step)

        step = step.astype(jnp.float32)
        t = step / num_steps
        t_next = (step + 1) / num_steps

        s = jax.lax.stop_gradient(s)

        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(dim)
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
            langevin = jax.lax.stop_gradient(langevin)
            # langevin = langevin * t  # ?
        elif partial_energy:
            log_prob = target.log_prob(s)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        if partial_energy:
            log_f = log_f + jax.lax.cond(
                t == 0.0,
                lambda _: 0.0,
                lambda args: get_partial_energy(args[0], args[1], args[2], args[3]),
                operand=(s, t, sigma_t, log_prob),
            )

        # Euler-Maruyama integration of the SDE
        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)

        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)

        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        shrink = (t_next - dt) / t_next  # == t / t_next when uniform time steps
        bwd_mean = shrink * s_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt)

        bwd_log_prob = jax.lax.cond(
            t == 0.0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Compute importance weight increment
        next_state = (s_next, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, log_f, s)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        sigma_t = sigmas(step)

        step = step.astype(jnp.float32)
        t = step / num_steps
        t_next = (step + 1) / num_steps

        s_next = jax.lax.stop_gradient(s_next)

        shrink = (t_next - dt) / t_next
        bwd_mean = shrink * s_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt)

        s, key_gen = jax.lax.cond(
            t == 0.0,
            lambda _: (jnp.zeros_like(s_next), key_gen),
            lambda args: sample_kernel(args[0], args[1], args[2]),
            operand=(key_gen, bwd_mean, bwd_scale),
        )
        s = jax.lax.stop_gradient(s)

        bwd_log_prob = jax.lax.cond(
            t == 0.0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Compute forward SDE components
        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(dim)
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
            langevin = jax.lax.stop_gradient(langevin)
            # langevin = langevin * t  # ?
        elif partial_energy:
            log_prob = target.log_prob(s)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        if partial_energy:
            log_f = log_f + jax.lax.cond(
                t == 0.0,
                lambda _: 0.0,
                lambda args: get_partial_energy(args[0], args[1], args[2], args[3]),
                operand=(s, t, sigma_t, log_prob),
            )

        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)

        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, log_f, s)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.zeros(dim)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(num_steps))
        final_x, _ = aux
    else:
        init_x = terminal_x
        if init_x is None:
            init_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )
        final_x = init_x

    fwd_log_probs, bwd_log_probs, log_fs, trajectories = per_step_output

    stochastic_costs = jnp.zeros_like(fwd_log_probs)
    terminal_costs = -target.log_prob(final_x)

    return (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        log_fs,
        trajectories,
    )


def rnd(
    key,
    model_state,
    params,
    reference_process,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_xs: Array | None = None,
):
    seeds = jax.random.split(key, num=batch_size)
    if reference_process == "pinned_brownian":
        (
            final_x,
            fwd_log_probs,
            bwd_log_probs,
            stochastic_costs,
            terminal_costs,
            log_fs,
            trajectories,
        ) = jax.vmap(
            per_sample_rnd_pinned_brownian,
            in_axes=(0, None, None, None, None, None, None, None, None, None, 0),
        )(
            seeds,
            model_state,
            params,
            aux_tuple,
            target,
            num_steps,
            noise_schedule,
            use_lp,
            partial_energy,
            prior_to_target,
            terminal_xs,
        )
        running_costs = fwd_log_probs - bwd_log_probs

    elif reference_process == "ou":
        # TODO
        raise NotImplementedError("OU reference process not implemented yet.")
    else:
        raise ValueError(f"Reference process {reference_process} not supported.")

    if not prior_to_target:
        running_costs = running_costs[:, ::-1]
        stochastic_costs = stochastic_costs[:, ::-1]
        log_fs = log_fs[:, ::-1]
        trajectories = trajectories[:, ::-1]

    log_fs = jnp.concatenate([log_fs, -terminal_costs[:, None]], axis=1)
    trajectories = jnp.concatenate([trajectories, final_x[:, None]], axis=1)

    return (
        final_x,
        running_costs.sum(1),
        stochastic_costs.sum(1),
        terminal_costs,
        running_costs,  # log_pfs - log_pbs
        log_fs,  # log_Fs
        trajectories,
    )


def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[
        [
            RandomKey,  # key
            TrainState,  # model_state
            ModelParams,  # params
            Array | None,  # terminal_xs
        ],
        tuple[Array, Array, Array, Array],
    ],
    n_chunks: int,
    terminal_xs: Array | None = None,
):
    aux = rnd_partial(key, model_state, params, terminal_xs=terminal_xs)
    _, log_pfs_over_pbs, _, terminal_costs, running_costs, log_fs, trajectories = aux

    db_discrepancy = log_fs[:, :-1] + running_costs - log_fs[:, 1:]

    bs, T = db_discrepancy.shape
    assert T % n_chunks == 0

    subtb_discrepancy = db_discrepancy.reshape(bs, n_chunks, -1).sum(-1)

    subtb_losses = jnp.square(subtb_discrepancy)
    return jnp.mean(subtb_losses.mean(-1)), (
        trajectories[:, T // n_chunks :: T // n_chunks],
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(tau)/pf(tau))
        jax.lax.stop_gradient(-subtb_discrepancy),  # log(f(s')pb(s'->s)/f(s)pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(subtb_losses),
    )
