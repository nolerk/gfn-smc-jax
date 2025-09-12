from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.dds.dds_rnd import cos_sq_fn_step_scheme


def sample_kernel(key_gen, mean, scale):
    key, key_gen = jax.random.split(key_gen)
    eps = jnp.clip(jax.random.normal(key, shape=(mean.shape[0],)), -4.0, 4.0)
    return mean + scale * eps, key_gen


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def per_sample_rnd_pinned_brownian(
    seed,
    model_state,
    params,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    dim = aux_tuple
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)

        step = step.astype(jnp.float32)
        t = (step / num_steps).astype(jnp.float32)
        t_next = (step + 1) / num_steps

        s = jax.lax.stop_gradient(s)

        # Compute SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
            # langevin = langevin * t  # ?
        else:
            langevin = jnp.zeros(dim)
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
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

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)

        step = step.astype(jnp.float32)
        t = (step / num_steps).astype(jnp.float32)
        t_next = (step + 1) / num_steps

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
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
            # langevin = langevin * t  # ?
        else:
            langevin = jnp.zeros(dim)
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s)
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

    fwd_log_probs, bwd_log_probs, trajectories = per_step_output
    stochastic_costs = jnp.zeros_like(fwd_log_probs)
    terminal_costs = -target.log_prob(final_x)

    return (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        trajectories,
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
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    # raise NotImplementedError("OU-DDS reference process has to be checked more thoroughly.")
    init_std, init_sampler, init_log_prob, noise_scale = aux_tuple
    betas = cos_sq_fn_step_scheme(num_steps)

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        sqrt_at = jnp.clip(noise_scale * jnp.sqrt(betas[step]), 0, 1)
        sqrt_1_minus_at = jnp.sqrt(1 - sqrt_at**2)

        s = jax.lax.stop_gradient(s)

        # Compute forward SDE components
        t = (step / num_steps).astype(jnp.float32)
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
            # langevin = langevin * t  # ?
        else:
            langevin = jnp.zeros(s.shape[0])
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)

        # Exponential integration of the SDE
        fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * model_output
        fwd_scale = sqrt_at * init_std
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        sqrt_at_next = jnp.clip(noise_scale * jnp.sqrt(betas[step]), 0, 1)
        sqrt_1_minus_at_next = jnp.sqrt(1 - sqrt_at_next**2)
        bwd_mean = sqrt_1_minus_at_next * s_next
        bwd_scale = sqrt_at_next * init_std
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        sqrt_at_next = jnp.clip(noise_scale * jnp.sqrt(betas[step]), 0, 1)
        sqrt_1_minus_at_next = jnp.sqrt(1 - sqrt_at_next**2)

        s_next = jax.lax.stop_gradient(s_next)

        # Compute backward SDE components
        bwd_mean = sqrt_1_minus_at_next * s_next
        bwd_scale = sqrt_at_next * init_std
        s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = jax.lax.cond(
            bwd_scale <= 0.0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Compute forward SDE components
        sqrt_at = jnp.clip(noise_scale * jnp.sqrt(betas[step]), 0, 1)
        sqrt_1_minus_at = jnp.sqrt(1 - sqrt_at**2)
        t = (step / num_steps).astype(jnp.float32)
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
            # langevin = langevin * t  # ?
        else:
            langevin = jnp.zeros(s.shape[0])
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * model_output
        fwd_scale = sqrt_at * init_std
        fwd_log_prob = jax.lax.cond(
            fwd_scale <= 0.0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(s_next, fwd_mean, fwd_scale),
        )

        # Return next state and per-step output
        next_state = (s, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, s)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        key, key_gen = jax.random.split(key_gen)
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

    fwd_log_probs, bwd_log_probs, trajectories = per_step_output
    fwd_log_probs = jnp.concatenate([init_log_prob(init_x)[None], fwd_log_probs])
    bwd_log_probs = jnp.concatenate([jnp.array(0.0)[None], bwd_log_probs])

    stochastic_costs = jnp.zeros_like(fwd_log_probs)
    terminal_costs = -target.log_prob(final_x)

    return (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        trajectories,
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
    prior_to_target=True,
    terminal_xs: Array | None = None,
):
    seeds = jax.random.split(key, num=batch_size)
    # Map reference process names to their corresponding functions
    process_functions = {
        "pinned_brownian": per_sample_rnd_pinned_brownian,
        "ou": per_sample_rnd_ou,
        "ou_dds": per_sample_rnd_ou_dds,
    }

    # Get the appropriate function and execute it
    per_sample_fn = process_functions[reference_process]
    (
        final_x,
        fwd_log_probs,
        bwd_log_probs,
        stochastic_costs,
        terminal_costs,
        trajectories,
    ) = jax.vmap(
        per_sample_fn,
        in_axes=(0, None, None, None, None, None, None, None, None, 0),
    )(
        seeds,
        model_state,
        params,
        aux_tuple,
        target,
        num_steps,
        noise_schedule,
        use_lp,
        prior_to_target,
        terminal_xs,
    )
    running_costs = fwd_log_probs - bwd_log_probs

    # trajectories = jnp.concatenate([trajectories, final_x[:, None]], axis=1)

    return final_x, running_costs.sum(1), stochastic_costs.sum(1), terminal_costs


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
    loss_type: Literal["tb", "lv"],
    terminal_xs: Array | None = None,
):
    aux = rnd_partial(key, model_state, params, terminal_xs=terminal_xs)
    final_x, running_costs, _, terminal_costs = aux
    log_ratio = running_costs + terminal_costs  # log_pfs - log_pbs - log_rewards
    if loss_type == "tb":
        losses = log_ratio + params["params"]["logZ"]
    else:  # loss_type == "lv"
        losses = log_ratio - jnp.mean(log_ratio)

    losses = jnp.square(losses)

    return jnp.mean(losses), (
        final_x,
        jax.lax.stop_gradient(-running_costs),  # log_pbs - log_pfs
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(losses),  # losses
    )
