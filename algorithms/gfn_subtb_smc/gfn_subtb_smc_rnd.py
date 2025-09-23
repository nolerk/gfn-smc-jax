from typing import Callable, Literal

import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.dds.dds_rnd import cos_sq_fn_step_scheme
from algorithms.gfn_tb.gfn_tb_rnd import sample_kernel, log_prob_kernel
from targets.base_target import Target


### per-step RND functions ###


def get_flow_bias(weight, ref_log_prob, log_prob):
    return (1 - weight) * ref_log_prob + weight * log_prob


def per_sample_subtraj_rnd_pinned_brownian(*args, **kwargs):
    raise NotImplementedError("Pinned Brownian reference process not implemented yet.")


def per_sample_subtraj_rnd_ou(*args, **kwargs):
    raise NotImplementedError("OU reference process not implemented yet.")


def per_sample_subtraj_rnd_ou_dds(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    input_state: Array,
    aux_tuple: tuple,
    target: Target,
    num_steps: int,
    timestep_tup: tuple[int, int],  # (start_step, subtraj_length)
    noise_schedule: Callable[[int], float],  # Not used here
    use_lp: bool,
    partial_energy: bool,
    prior_to_target: bool = True,
):
    init_std, init_log_prob, noise_scale = aux_tuple
    alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale)
    lambda_ts = 1 - (1 - alphas)[::-1].cumprod()[::-1]

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        t = step / num_steps

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
        t = step / num_steps

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
    start_step, traj_length = timestep_tup
    if prior_to_target:
        start_state = input_state
        aux = (start_state, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, start_step + jnp.arange(traj_length)
        )
        end_state, _ = aux
    else:
        end_state = input_state
        aux = (end_state, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, (start_step + jnp.arange(traj_length))[::-1]
        )
        start_state, _ = aux

    subtrajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    return end_state, subtrajectory, fwd_log_prob, bwd_log_prob, log_f


def get_end_state_log_f(
    model_state: TrainState,
    params: ModelParams,
    end_state: Array,
    aux_tuple: tuple,
    target: Target,
    num_steps: int,
    end_step: int,
    noise_schedule: Callable[[int], float],  # Not used here
    partial_energy: bool,
):
    init_std, init_log_prob, noise_scale = aux_tuple
    alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale)
    lambda_ts = 1 - (1 - alphas)[::-1].cumprod()[::-1]

    _, end_state_log_f = model_state.apply_fn(
        params, end_state, (end_step / num_steps) * jnp.ones(1), jnp.zeros(end_state.shape[0])
    )  # langevin doesn't affect the log_f
    if partial_energy:
        weight = 1 - lambda_ts[end_step]
        end_state_log_f = end_state_log_f + get_flow_bias(
            weight, init_log_prob(end_state), target.log_prob(end_state)
        )
    return end_state_log_f


### simulate ###
def batch_simulate_subtraj_fwd(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    batch_size: int,
    initial_dist: distrax.Distribution | None,
    target: Target,
    num_steps: int,
    num_subtrajs: int,
    sampling_configs: tuple[
        Literal["pinned_brownian", "ou", "ou_dds"],  # reference_process
        tuple,  # aux_tuple; for ou_dds, (init_std, initial_dist.log_prob, noise_scale)
        Callable[[int], float],  # noise_schedule
        bool,  # use_lp
        bool,  # partial_energy
    ],
    smc_settings: dict,  # resampling confg
    mcmc_settings: dict,  # mcmc config
):
    assert num_steps % num_subtrajs == 0
    subtraj_length = num_steps // num_subtrajs

    # TODO: support mcmc
    markov_kernel = None

    ### subtrajectory step functions ###
    def batch_simulate_subtrajectory(step_input, per_step_input):
        (states, log_iws, key_gen) = step_input
        start_step = per_step_input

        reference_process, aux_tuple, noise_schedule, use_lp, partial_energy = sampling_configs

        ## vectorized subtrajectory sampling
        key, key_gen = jax.random.split(key_gen)
        keys = jax.random.split(key, num=batch_size)
        per_sample_fn = {
            "pinned_brownian": per_sample_subtraj_rnd_pinned_brownian,
            "ou": per_sample_subtraj_rnd_ou,
            "ou_dds": per_sample_subtraj_rnd_ou_dds,
        }[reference_process]

        next_states, subtrajectories, fwd_log_probs, bwd_log_probs, log_fs = jax.vmap(
            per_sample_fn,
            in_axes=(0, None, None, 0, None, None, None, None, None, None, None, None),
        )(
            keys,
            model_state,
            params,
            states,
            aux_tuple,
            target,
            num_steps,
            (start_step, subtraj_length),
            noise_schedule,
            use_lp,
            partial_energy,
            True,  # prior_to_target
        )

        def _compute_last_step(_):
            end_state_log_fs = target.log_prob(next_states)
            log_rewards = end_state_log_fs
            return end_state_log_fs, log_rewards

        def _compute_not_last_step(_):
            end_state_log_fs = jax.vmap(
                get_end_state_log_f,
                in_axes=(None, None, 0, None, None, None, None, None, None),
            )(
                model_state,
                params,
                next_states,
                aux_tuple,
                target,
                num_steps,
                start_step + subtraj_length,
                noise_schedule,
                partial_energy,
            )
            log_rewards = jnp.zeros(batch_size)
            return end_state_log_fs, log_rewards

        end_state_log_fs, log_rewards = jax.lax.cond(
            jnp.equal(start_step + subtraj_length, num_steps),
            _compute_last_step,
            _compute_not_last_step,
            next_states,
        )

        next_log_iws = log_iws + (
            end_state_log_fs + bwd_log_probs.sum(-1) - log_fs[:, 0] + fwd_log_probs.sum(-1)
        )

        ## Do resampling with the adaptive tempering
        # TODO

        ## Do MCMC
        # TODO

        ## Return outputs
        next_step_input = (next_states, next_log_iws, key_gen)
        per_subtraj_outputs = (subtrajectories, fwd_log_probs, bwd_log_probs, log_fs, log_rewards)
        return next_step_input, per_subtraj_outputs

    # Define initial state
    key, key_gen = jax.random.split(key)
    if initial_dist is not None:
        init_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        init_states = jnp.zeros((batch_size, target.dim))
    init_fwd_log_probs = (
        initial_dist.log_prob(init_states) if initial_dist is not None else jnp.zeros(batch_size)
    )
    init_input = (init_states, -init_fwd_log_probs, key_gen)

    # Define per step inputs
    subtraj_start_steps = jnp.arange(0, num_steps, subtraj_length)
    per_subtraj_inputs = subtraj_start_steps

    final_outputs, per_subtraj_outputs = jax.lax.scan(
        batch_simulate_subtrajectory, init_input, per_subtraj_inputs
    )

    final_states, final_iws, _ = final_outputs
    # final_states.shape == (batch_size, dim)
    # final_iws.shape == (batch_size,)
    subtrajectories, fwd_log_probs, bwd_log_probs, log_fs, log_rewards = per_subtraj_outputs
    # subtrajectories.shape == (#subtrajs, batch_size, subtraj_length, dim)
    # log_rewards.shape == (#subtrajs, batch_size)
    # other outputs have shape (#subtrajs, batch_size, subtraj_length)

    return (
        final_states,
        subtrajectories,
        fwd_log_probs,
        bwd_log_probs,
        log_fs,
        log_rewards,
        init_fwd_log_probs,
    )


####################################################################################################
# Temporary loss function for sanity check
def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    simulate_subtraj: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    loss_type: Literal["subtb", "tb_subtb", "lv_subtb"] = "subtb",
    n_chunks: int = 1,
    subtb_weight: float = 1.0,
    invtemp: float = 1.0,
    huber_delta: float = 1e5,
    logr_clip: float = -1e5,
):
    (
        final_states,
        subtrajectories,
        fwd_log_probs,
        bwd_log_probs,
        log_fs,
        log_rewards,
        init_fwd_log_probs,
    ) = simulate_subtraj(key, model_state, params)

    n_subtrajs, bs, subtraj_length, dim = subtrajectories.shape
    T = n_subtrajs * subtraj_length

    trajectories = jnp.transpose(subtrajectories, (1, 0, 2, 3)).reshape(bs, T, dim)
    fwd_log_probs = jnp.transpose(fwd_log_probs, (1, 0, 2)).reshape(bs, T)
    bwd_log_probs = jnp.transpose(bwd_log_probs, (1, 0, 2)).reshape(bs, T)
    log_fs = jnp.transpose(log_fs, (1, 0, 2)).reshape(bs, T)
    log_rewards = log_rewards[-1, :]

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    log_rewards = jnp.where(
        log_rewards > logr_clip,
        log_rewards,
        logr_clip - jnp.log(logr_clip - log_rewards),
    )
    logZ = params["params"]["logZ"]

    tb_losses = jnp.zeros_like(log_rewards)
    if loss_type != "subtb":
        log_ratio = (init_fwd_log_probs + log_pfs_over_pbs.sum(-1)) - log_rewards * invtemp
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
    log_fs = jnp.concatenate([log_fs, (log_rewards * invtemp)[:, None]], axis=1)

    # db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # subtb_discrepancy1 = db_discrepancy.reshape(bs, n_chunks, -1).sum(-1)
    # The below is equivalent to the above lines but avoids numerical instability.
    log_pfs_over_pbs = jax.lax.stop_gradient(log_pfs_over_pbs)
    subtb_discrepancy1 = (
        log_fs[:, :-1:subtraj_length]
        + log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1)
        - log_fs[:, subtraj_length::subtraj_length]
    )

    log_pfs_over_pbs_cumsum = jnp.cumsum(log_pfs_over_pbs[:, ::-1], axis=-1)[:, ::-1]
    subtb_discrepancy2 = (
        log_fs[:, :-1:subtraj_length]
        + log_pfs_over_pbs_cumsum[:, ::subtraj_length]
        - log_fs[:, [-1]]
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
    return jnp.mean(tb_losses + subtb_weight * subtb_losses.mean(-1)), (
        trajectories,
        -log_pfs_over_pbs,  # log(pb(s'->s)/pf(s->s'))
        log_rewards,
        jax.lax.stop_gradient(tb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )


####################################################################################################
