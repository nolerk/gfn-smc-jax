from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams


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
    sde_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    dim, _ = sde_tuple
    target_log_prob = target.log_prob

    sigmas = noise_schedule
    scaled_target_log_prob = lambda x, t: t * target_log_prob(x)
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        x, key_gen = state
        step = per_step_input

        t = step / num_steps
        t_next = (step + 1) / num_steps
        sigma_t = sigmas(step)

        step = step.astype(jnp.float32)
        x = jax.lax.stop_gradient(x)

        # Compute SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(scaled_target_log_prob)(x, t))
        else:
            langevin = jnp.zeros(dim)
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)

        x_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        x_next = jax.lax.stop_gradient(x_next)

        fwd_log_prob = log_prob_kernel(x_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        shrink = (t_next - dt) / t_next  # == t / t_next when uniform time steps
        bwd_mean = shrink * x_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt) + 1e-8

        bwd_log_prob = jax.lax.cond(
            t == 0.0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(x, bwd_mean, bwd_scale),
        )

        # Compute importance weight increment
        next_state = (x_next, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, x_next)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        x_next, key_gen = state
        step = per_step_input

        t = step / num_steps
        t_next = (step + 1) / num_steps
        sigma_t = sigmas(step)

        step = step.astype(jnp.float32)
        x_next = jax.lax.stop_gradient(x_next)

        shrink = (t_next - dt) / t_next
        bwd_mean = shrink * x_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt) + 1e-8

        x, key_gen = jax.lax.cond(
            t == 0.0,
            lambda _: (jnp.zeros_like(x_next), key_gen),
            lambda args: sample_kernel(args[0], args[1], args[2]),
            operand=(key_gen, bwd_mean, bwd_scale),
        )
        x = jax.lax.stop_gradient(x)

        bwd_log_prob = jax.lax.cond(
            t == 0.0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(args[0], args[1], args[2]),
            operand=(x, bwd_mean, bwd_scale),
        )

        # Compute forward SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(scaled_target_log_prob)(x, t))
        else:
            langevin = jnp.zeros(dim)
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        fwd_mean = x + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)

        fwd_log_prob = log_prob_kernel(x_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (x, key_gen)
        per_step_output = (fwd_log_prob, bwd_log_prob, x)
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

    fwd_log_prob, bwd_log_prob, x_t = per_step_output
    stochastic_costs = jnp.zeros_like(fwd_log_prob)
    terminal_cost = -target_log_prob(final_x)

    return final_x, fwd_log_prob, bwd_log_prob, stochastic_costs, terminal_cost


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
    prior_to_target=True,
    terminal_xs: Array | None = None,
):
    seeds = jax.random.split(key, num=batch_size)
    if reference_process == "pinned_brownian":
        final_x, fwd_log_probs, bwd_log_probs, stochastic_costs, terminal_costs = jax.vmap(
            per_sample_rnd_pinned_brownian,
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

    elif reference_process == "ou":
        # TODO
        raise NotImplementedError("OU reference process not implemented yet.")
    else:
        raise ValueError(f"Reference process {reference_process} not supported.")

    return final_x, running_costs.sum(1), stochastic_costs.sum(1), terminal_costs


def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[
        [RandomKey, TrainState, ModelParams, Array | None],  # key, model_state, params, terminal_xs
        tuple[Array, Array, Array, Array],
    ],
    loss_type: Literal["tb", "lv"],
    terminal_xs: Array | None = None,
):
    aux = rnd_partial(key, model_state, params, terminal_xs=terminal_xs)
    samples, running_costs, _, terminal_costs = aux
    log_ratio = running_costs + terminal_costs
    if loss_type == "tb":
        losses = log_ratio + params["params"]["logZ"]
    else:  # loss_type == "lv"
        losses = log_ratio - jnp.mean(log_ratio)

    losses = jnp.square(losses)

    return jnp.mean(losses), (
        -running_costs,  # log_pbs / log_pfs
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(losses),  # losses
        samples,
    )
