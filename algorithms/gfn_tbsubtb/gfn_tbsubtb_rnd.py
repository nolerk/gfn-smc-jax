from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams


def loss_fn_joint(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    loss_type: Literal["tb", "lv"] = "tb",
    n_chunks: int = 1,
    subtb_weight: float = 0.1,
    invtemp: float = 1.0,
    huber_delta: float = 1e5,
    logr_clip: float = -1e5,
):
    (
        _,
        running_costs,
        _,
        terminal_costs,
        log_pfs_over_pbs,
        trajectories,
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
    log_ratio = running_costs - log_rewards * invtemp  # running_costs already contains initial pfs
    logZ = params["params"]["logZ"]
    if loss_type == "tb":
        tb_losses = log_ratio + logZ
        logZ = jax.lax.stop_gradient(logZ)
    else:  # loss_type == "lv"
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
        log_pfs_over_pbs,
        trajectories,
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
):
    (
        _,
        _,
        _,
        terminal_costs,
        log_pfs_over_pbs,
        trajectories,
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

    log_fs = log_fs.at[:, 0].set(jax.lax.stop_gradient(logZ) + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

    # We only want to optimize log_fs; gradients should not flow through log_pfs_over_pbs.
    log_pfs_over_pbs = jax.lax.stop_gradient(log_pfs_over_pbs)

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
