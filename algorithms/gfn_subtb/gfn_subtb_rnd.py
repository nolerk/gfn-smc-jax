from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.dds.dds_rnd import cos_sq_fn_step_scheme
from algorithms.gfn_tb.gfn_tb_rnd import sample_kernel, log_prob_kernel


def get_flow_bias(weight, ref_log_prob, log_prob):
    return (1 - weight) * ref_log_prob + weight * log_prob


def ref_log_prob_pinned_brownian(s, t, sigma_t):
    ref_log_var = jnp.log((sigma_t**2) * jnp.array(t))
    log_p_ref = -0.5 * (
        jnp.log(2 * jnp.pi) + ref_log_var + jnp.exp(-ref_log_var) * jnp.square(s)
    ).sum(-1)
    return log_p_ref


def per_sample_rnd_pinned_brownian(
    key,
    model_state,
    params: ModelParams,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_x: Array | None = None,
    log_reward: Array | None = None,
):
    dim = aux_tuple
    dt = 1.0 / num_steps

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
            # langevin = langevin * t  # ?
        elif partial_energy:
            log_prob = target.log_prob(s)

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            ref_log_prob = jax.lax.cond(
                step == 0,
                lambda _: 0.0,
                lambda args: ref_log_prob_pinned_brownian(args[0], args[1], args[2]),
                operand=(s, t, sigma_t),
            )
            log_f_bias = get_flow_bias(t, ref_log_prob, log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

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
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
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
            # langevin = langevin * t  # ?
        elif partial_energy:
            log_prob = target.log_prob(s)

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            ref_log_prob = jax.lax.cond(
                step == 0,
                lambda _: 0.0,
                lambda args: ref_log_prob_pinned_brownian(args[0], args[1], args[2]),
                operand=(s, t, sigma_t),
            )
            log_f_bias = get_flow_bias(t, ref_log_prob, log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    key, key_gen = jax.random.split(key)
    if prior_to_target:
        init_x = jnp.zeros(dim)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(num_steps))
        terminal_x, _ = aux
        trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    else:
        if terminal_x is None:
            terminal_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )
        trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
        trajectory = trajectory[::-1]
        fwd_log_prob = fwd_log_prob[::-1]
        bwd_log_prob = bwd_log_prob[::-1]
        log_f = log_f[::-1]

    if log_reward is None:
        log_reward = target.log_prob(terminal_x)

    trajectory = jnp.concatenate([trajectory, terminal_x[None]], axis=0)
    log_f = jnp.concatenate([log_f, log_reward[None]], axis=0)

    return trajectory, fwd_log_prob, bwd_log_prob, log_f, log_reward, jnp.array(0.0)


def per_sample_rnd_ou(
    key,
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
    log_reward: Array | None = None,
):
    raise NotImplementedError("OU reference process not implemented yet.")


def per_sample_rnd_ou_dds(
    key,
    model_state,
    params: ModelParams,
    aux_tuple,
    target,
    num_steps,
    noise_schedule,  # Not used here
    use_lp,
    partial_energy,
    prior_to_target=True,
    terminal_x: Array | None = None,
    log_reward: Array | None = None,
):
    init_std, init_sampler, init_log_prob, noise_scale = aux_tuple
    alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale)
    lambda_ts = 1 - (1 - alphas)[::-1].cumprod()[::-1]

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

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            # weight = jnp.sqrt((1 - alphas)[step:].prod())
            weight = 1 - lambda_ts[step]
            log_f_bias = get_flow_bias(weight, init_log_prob(s), log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

        # Exponential integration of the SDE
        sqrt_at = jnp.clip(jnp.sqrt(alphas[step]), 0, 1)
        sqrt_1_minus_at = jnp.sqrt(1 - sqrt_at**2)
        fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * model_output
        fwd_scale = sqrt_at * init_std
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        bwd_mean = sqrt_1_minus_at * s_next
        bwd_scale = sqrt_at * init_std
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        t = (step / num_steps).astype(jnp.float32)

        s_next = jax.lax.stop_gradient(s_next)

        # Compute backward SDE components
        sqrt_at_next = jnp.clip(jnp.sqrt(alphas[step]), 0, 1)
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

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            # weight = jnp.sqrt((1 - alphas)[step:].prod())
            weight = 1 - lambda_ts[step]
            log_f_bias = get_flow_bias(weight, init_log_prob(s), log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

        fwd_mean = sqrt_1_minus_at_next * s + sqrt_at_next**2 * model_output
        fwd_scale = sqrt_at_next * init_std
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Return next state and per-step output
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    key, key_gen = jax.random.split(key)
    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(num_steps))
        terminal_x, _ = aux
        trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    else:
        if terminal_x is None:
            terminal_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )
        init_x, _ = aux
        trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
        trajectory = trajectory[::-1]
        fwd_log_prob = fwd_log_prob[::-1]
        bwd_log_prob = bwd_log_prob[::-1]
        log_f = log_f[::-1]

    init_fwd_log_prob = init_log_prob(init_x)
    if log_reward is None:
        log_reward = target.log_prob(terminal_x)

    trajectory = jnp.concatenate([trajectory, terminal_x[None]], axis=0)
    log_f = jnp.concatenate([log_f, log_reward[None]], axis=0)

    return trajectory, fwd_log_prob, bwd_log_prob, log_f, log_reward, init_fwd_log_prob


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
    log_rewards: Array | None = None,
):
    keys = jax.random.split(key, num=batch_size)
    per_sample_fn = {
        "pinned_brownian": per_sample_rnd_pinned_brownian,
        "ou": per_sample_rnd_ou,
        "ou_dds": per_sample_rnd_ou_dds,
    }[reference_process]

    (
        trajectories,
        fwd_log_probs,
        bwd_log_probs,
        log_fs,
        log_rewards,
        init_fwd_log_probs,
    ) = jax.vmap(
        per_sample_fn,
        in_axes=(0, None, None, None, None, None, None, None, None, None, 0, 0),
    )(
        keys,
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
        log_rewards,
    )

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories[:, -1],
        log_pfs_over_pbs.sum(1) + init_fwd_log_probs,
        jnp.zeros_like(log_rewards),
        -log_rewards,
        trajectories,
        log_pfs_over_pbs,  # log_pfs - log_pbs
        log_fs,
        init_fwd_log_probs,
    )


def loss_fn_tb(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    loss_type: Literal["tb", "lv"] = "tb",
    invtemp: float = 1.0,
    huber_delta: float = 1e5,
    logr_clip: float = -1e5,
):
    (
        _,
        running_costs,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        _,
        _,
    ) = rnd_partial(key, model_state, params)
    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    log_ratio = running_costs - log_rewards * invtemp  # running_costs already contains initial pfs
    if loss_type == "tb":
        losses = log_ratio + params["params"]["logZ"]
    else:  # loss_type == "lv"
        losses = log_ratio - jnp.mean(log_ratio)

    losses = jnp.where(
        jnp.abs(losses) <= huber_delta,
        jnp.square(losses),
        huber_delta * jnp.abs(losses) - 0.5 * huber_delta**2,
    )

    return jnp.mean(losses), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(losses),
        jnp.zeros_like(losses),
    )


def loss_fn_subtb(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    n_chunks: int,
    invtemp: float = 1.0,
    huber_delta: float = 1e5,
    logr_clip: float = -1e5,
    flow_only: bool = False,  # if True, only optimize flow
):
    (
        _,
        _,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
        init_fwd_log_probs,
    ) = rnd_partial(key, model_state, params)

    bs, T = log_pfs_over_pbs.shape
    assert T % n_chunks == 0
    chunk_size = T // n_chunks

    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    logZ = params["params"]["logZ"]
    if flow_only:
        logZ = jax.lax.stop_gradient(logZ)
        log_pfs_over_pbs = jax.lax.stop_gradient(log_pfs_over_pbs)

    log_fs = log_fs.at[:, 0].set(logZ + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

    # db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # subtb_discrepancy1 = db_discrepancy.reshape(bs, n_chunks, -1).sum(-1)
    # The below is equivalent to the above lines but avoids numerical instability.
    subtb_discrepancy1 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1)
        - log_fs[:, chunk_size::chunk_size]
    )

    log_pfs_over_pbs_cumsum = jnp.cumsum(log_pfs_over_pbs[:, ::-1], axis=-1)[:, ::-1]
    subtb_discrepancy2 = (
        log_fs[:, :-1:chunk_size] + log_pfs_over_pbs_cumsum[:, ::chunk_size] - log_fs[:, [-1]]
    ) / jnp.arange(1, n_chunks + 1)[None, ::-1]

    # subtb_discrepancy = subtb_discrepancy1
    # subtb_discrepancy = subtb_discrepancy2
    subtb_discrepancy = jnp.concatenate([subtb_discrepancy1, subtb_discrepancy2], axis=1)

    subtb_losses = jnp.where(
        jnp.abs(subtb_discrepancy) <= huber_delta,
        jnp.square(subtb_discrepancy),
        huber_delta * jnp.abs(subtb_discrepancy) - 0.5 * huber_delta**2,
    )

    log_pfs_over_pbs = log_pfs_over_pbs.at[:, 0].set(log_pfs_over_pbs[:, 0] + init_fwd_log_probs)
    return jnp.mean(subtb_losses.mean(-1)), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jnp.zeros_like(subtb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )


def loss_fn_joint(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    loss_type: Literal["subtb", "tb_subtb", "lv_subtb"] = "subtb",
    n_chunks: int = 1,
    subtb_weight: float = 1.0,
    invtemp: float = 1.0,
    huber_delta: float = 1e5,
    logr_clip: float = -1e5,
):
    (
        _,
        running_costs,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
        init_fwd_log_probs,
    ) = rnd_partial(key, model_state, params)

    bs, T = log_pfs_over_pbs.shape
    assert T % n_chunks == 0
    chunk_size = T // n_chunks

    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    logZ = params["params"]["logZ"]

    tb_losses = jnp.zeros_like(log_rewards)
    if loss_type != "subtb":
        log_ratio = (
            running_costs - log_rewards * invtemp
        )  # running_costs already contains initial pfs
        if loss_type == "tb_subtb":
            tb_losses = log_ratio + logZ
            logZ = jax.lax.stop_gradient(logZ)
        else:  # loss_type == "lv_subtb"
            tb_losses = log_ratio - jnp.mean(log_ratio)

        tb_losses = jnp.where(
            jnp.abs(tb_losses) <= huber_delta,
            jnp.square(tb_losses),
            huber_delta * jnp.abs(tb_losses) - 0.5 * huber_delta**2,
        )

    log_fs = log_fs.at[:, 0].set(logZ + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

    # db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # subtb_discrepancy1 = db_discrepancy.reshape(bs, n_chunks, -1).sum(-1)
    # The below is equivalent to the above lines but avoids numerical instability.
    subtb_discrepancy1 = (
        log_fs[:, :-1:chunk_size]
        + jax.lax.stop_gradient(log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1))
        - log_fs[:, chunk_size::chunk_size]
    )

    log_pfs_over_pbs_cumsum = jax.lax.stop_gradient(
        jnp.cumsum(log_pfs_over_pbs[:, ::-1], axis=-1)[:, ::-1]
    )
    subtb_discrepancy2 = (
        log_fs[:, :-1:chunk_size] + log_pfs_over_pbs_cumsum[:, ::chunk_size] - log_fs[:, [-1]]
    ) / jnp.arange(1, n_chunks + 1)[None, ::-1]

    # subtb_discrepancy = subtb_discrepancy1
    # subtb_discrepancy = subtb_discrepancy2
    subtb_discrepancy = jnp.concatenate([subtb_discrepancy1, subtb_discrepancy2], axis=1)

    subtb_losses = jnp.where(
        jnp.abs(subtb_discrepancy) <= huber_delta,
        jnp.square(subtb_discrepancy),
        huber_delta * jnp.abs(subtb_discrepancy) - 0.5 * huber_delta**2,
    )
    return jnp.mean(tb_losses + subtb_weight * subtb_losses.mean(-1)), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(tb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )
