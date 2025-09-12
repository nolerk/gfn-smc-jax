from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.dds.dds_rnd import cos_sq_fn_step_scheme
from algorithms.gfn_tb.gfn_tb_rnd import sample_kernel, log_prob_kernel


def per_sample_rnd_pinned_brownian(
    seed,
    model_state,
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    dim = aux_tuple
    dt = 1.0 / num_steps

    def get_partial_energy(s, t, sigma_t, log_prob):
        ref_log_var = jnp.log((sigma_t**2) * jnp.array(t))
        log_p_ref = -0.5 * (
            jnp.log(2 * jnp.pi) + ref_log_var + jnp.exp(-ref_log_var) * jnp.square(s)
        ).sum(-1)
        return (1 - t) * log_p_ref + t * log_prob

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s = jax.lax.stop_gradient(s)

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
                step == 0,
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
            step == 0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Compute importance weight increment
        next_state = (s_next, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s, log_f)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s_next = jax.lax.stop_gradient(s_next)

        # Compute backward SDE components
        shrink = (t_next - dt) / t_next
        bwd_mean = shrink * s_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt)
        s, key_gen = jax.lax.cond(
            step == 0,
            lambda _: (jnp.zeros_like(s_next), key_gen),
            lambda args: sample_kernel(args[0], args[1], args[2]),
            operand=(key_gen, bwd_mean, bwd_scale),
        )
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = jax.lax.cond(
            step == 0,
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
                step == 0,
                lambda _: 0.0,
                lambda args: get_partial_energy(args[0], args[1], args[2], args[3]),
                operand=(s, t, sigma_t, log_prob),
            )

        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s, log_f)
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

    fwd_log_probs, bwd_log_probs, trajectories, log_fs = per_step_output
    stochastic_costs = jnp.zeros_like(fwd_log_probs)
    terminal_costs = -target.log_prob(final_x)

    return (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        trajectories,
        log_fs,
    )


def per_sample_rnd_ou(
    seed,
    model_state,
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    raise NotImplementedError("OU reference process not implemented yet.")


def per_sample_rnd_ou_dds(
    seed,
    model_state,
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,  # Not used here
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    init_std, init_sampler, init_log_prob, noise_scale = aux_tuple
    betas = cos_sq_fn_step_scheme(num_steps)

    def get_partial_energy(s, t, log_prob):
        return (1 - t) * init_log_prob(s) + t * log_prob

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        t = (step / num_steps).astype(jnp.float32)

        s = jax.lax.stop_gradient(s)

        # Compute forward SDE components
        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(s.shape[0])
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
            # langevin = langevin * t  # ?
        elif partial_energy:
            log_prob = target.log_prob(s)
        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        if partial_energy:
            log_f = log_f + get_partial_energy(s, t, log_prob)

        # Exponential integration of the SDE
        sqrt_at = jnp.clip(noise_scale * jnp.sqrt(betas[step - 1]), 0, 1)
        sqrt_1_minus_at = jnp.sqrt(1 - sqrt_at**2)
        fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * model_output
        fwd_scale = sqrt_at * init_std
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        sqrt_at_next = jnp.clip(
            noise_scale * jnp.sqrt(betas[step - 1]), 0, 1  # shouldn't we use step + 1?
        )
        sqrt_1_minus_at_next = jnp.sqrt(1 - sqrt_at_next**2)
        bwd_mean = sqrt_1_minus_at_next * s_next
        bwd_scale = sqrt_at_next * init_std
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s, log_f)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        t = (step / num_steps).astype(jnp.float32)

        s_next = jax.lax.stop_gradient(s_next)

        # Compute backward SDE components
        sqrt_at_next = jnp.clip(
            noise_scale * jnp.sqrt(betas[step - 1]), 0, 1  # shouldn't we use step + 1?
        )
        sqrt_1_minus_at_next = jnp.sqrt(1 - sqrt_at_next**2)
        bwd_mean = sqrt_1_minus_at_next * s_next
        bwd_scale = sqrt_at_next * init_std
        s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        # Compute forward SDE components
        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(s.shape[0])
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
            # langevin = langevin * t  # ?
        elif partial_energy:
            log_prob = target.log_prob(s)
        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        if partial_energy:
            log_f = log_f + get_partial_energy(s, t, log_prob)

        sqrt_at = jnp.clip(noise_scale * jnp.sqrt(betas[step - 1]), 0, 1)
        sqrt_1_minus_at = jnp.sqrt(1 - sqrt_at**2)
        fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * model_output
        fwd_scale = sqrt_at * init_std
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Return next state and per-step output
        next_state = (s, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s, log_f)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(1, num_steps))
        final_x, _ = aux
    else:
        init_x = terminal_x
        if init_x is None:
            init_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(1, num_steps)[::-1]
        )
        final_x = init_x

    fwd_log_probs, bwd_log_probs, trajectories, log_fs = per_step_output
    fwd_log_probs = jnp.concatenate([init_log_prob(init_x)[None], fwd_log_probs])
    bwd_log_probs = jnp.concatenate([jnp.array(0.0)[None], bwd_log_probs])
    _, log_f_init = model_state.apply_fn(
        params, jnp.zeros(init_x.shape[0]), 0.0 * jnp.ones(1), jnp.zeros(init_x.shape[0])
    )
    log_fs = jnp.concatenate([log_f_init[None], log_fs])

    stochastic_costs = jnp.zeros_like(fwd_log_probs)
    terminal_costs = -target.log_prob(final_x)

    return (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        trajectories,
        log_fs,
    )


def rnd(
    key,
    model_state,
    params,
    reference_process: Literal["pinned_brownian", "ou", "ou_dds"],
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
    per_sample_fn = {
        "pinned_brownian": per_sample_rnd_pinned_brownian,
        "ou": per_sample_rnd_ou,
        "ou_dds": per_sample_rnd_ou_dds,
    }[reference_process]

    (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        trajectories,
        log_fs,
    ) = jax.vmap(
        per_sample_fn,
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

    if not prior_to_target:
        running_costs = running_costs[:, ::-1]
        stochastic_costs = stochastic_costs[:, ::-1]
        trajectories = trajectories[:, ::-1]
        log_fs = log_fs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, final_x[:, None]], axis=1)
    log_fs = jnp.concatenate([log_fs, -terminal_costs[:, None]], axis=1)

    return (
        final_x,
        running_costs.sum(1),
        stochastic_costs.sum(1),
        terminal_costs,
        running_costs,  # log_pfs - log_pbs
        trajectories,
        log_fs,  # log_Fs
    )


def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, Array, Array, Array]],
    n_chunks: int,
):
    aux = rnd_partial(key, model_state, params)
    _, log_pfs_over_pbs, _, terminal_costs, running_costs, trajectories, log_fs = aux

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
