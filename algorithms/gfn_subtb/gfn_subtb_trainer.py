"""
Code for Sub-Trajectory Balance (SubTB) training (Note: SubTB generalizes DB and TB).
For further details see: https://arxiv.org/abs/2209.12782 and https://arxiv.org/abs/2501.06148
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.gfn_subtb.buffer import build_intermediate_state_buffer
from algorithms.gfn_subtb.gfn_subtb_rnd import rnd, loss_fn
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def gfn_subtb_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define initial and target density
    if alg_cfg.reference_process == "pinned_brownian":  # Following PIS
        aux_tuple = (dim,)
    elif alg_cfg.reference_process in ["ou", "ou_dds"]:  # DIS or DDS
        initial_dist = distrax.MultivariateNormalDiag(
            jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
        )
        aux_tuple = (alg_cfg.init_std, initial_dist.sample, initial_dist.log_prob)
        if alg_cfg.reference_process == "ou_dds":
            aux_tuple = (*aux_tuple, alg_cfg.noise_scale)  # DDS
    else:
        raise ValueError(f"Reference process {alg_cfg.reference_process} not supported.")

    # Initialize the buffer
    use_buffer = alg_cfg.buffer.use_buffer
    buffer = buffer_state = None
    if use_buffer:
        buffer = build_intermediate_state_buffer(
            dim=dim,
            max_length=alg_cfg.buffer.max_length_in_batches * alg_cfg.batch_size,
            prioritize_by=alg_cfg.buffer.prioritize_by,
            target_ess=alg_cfg.buffer.target_ess,
            sampling_method=alg_cfg.buffer.sampling_method,
            rank_k=alg_cfg.buffer.rank_k,
        )
        buffer_state = buffer.init(dtype=target_xs.dtype, device=target_xs.device)

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    rnd_partial_base = partial(
        rnd,
        reference_process=alg_cfg.reference_process,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=cfg.algorithm.num_steps,
        noise_schedule=cfg.algorithm.noise_schedule,
        use_lp=alg_cfg.model.use_lp,
        partial_energy=alg_cfg.partial_energy,
    )
    loss_fn_base = partial(loss_fn, n_chunks=alg_cfg.n_chunks)

    # Define the function to be JIT-ed for FWD pass
    @jax.jit
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params):
        # prior_to_target=True, terminal_xs=None
        rnd_p = partial(rnd_partial_base, batch_size=alg_cfg.batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_p)

    # --- Define the function to be JIT-ed for BWD pass ---
    @jax.jit
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_bwd_grad_fn(key, model_state, params, terminal_xs):
        # prior_to_target=False, terminal_xs is now an argument
        rnd_p = partial(
            rnd_partial_base,
            batch_size=alg_cfg.batch_size,
            prior_to_target=False,
            terminal_xs=terminal_xs,
        )
        return loss_fn_base(key, model_state, params, rnd_p)

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(rnd_partial_base, batch_size=cfg.eval_samples), target, target_xs, cfg
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    ### Prefill phase
    if use_buffer and alg_cfg.buffer.prefill_steps > 0:
        # Define the function to be JIT-ed for FWD pass
        @jax.jit
        def loss_fwd_nograd_fn(key, model_state, params):
            # prior_to_target=True, terminal_xs=None
            rnd_p = partial(rnd_partial_base, batch_size=alg_cfg.batch_size, prior_to_target=True)
            return loss_fn_base(key, model_state, params, rnd_p)

        assert buffer is not None and buffer_state is not None
        for _ in range(alg_cfg.buffer.prefill_steps):
            key, key_gen = jax.random.split(key_gen)
            _, (
                trajectories,
                log_pbs_over_pfs,
                _,  # subtb_discrepancy
                log_rewards,
                subtb_losses,
            ) = loss_fwd_nograd_fn(key, model_state, model_state.params)

            buffer_state = buffer.add(
                buffer_state,
                trajectories[:, -1],
                log_pbs_over_pfs,
                log_rewards,
                subtb_losses.sum(-1),
            )

    ### Training phase
    for it in range(alg_cfg.iters):

        # On-policy training with forward samples
        if not use_buffer or it % (alg_cfg.buffer.bwd_to_fwd_ratio + 1) == 0:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (
                trajectories,
                log_pbs_over_pfs,
                _,  # subtb_discrepancy
                log_rewards,
                subtb_losses,
            ) = loss_fwd_grad_fn(key, model_state, model_state.params)
            model_state = model_state.apply_gradients(grads=grads)

            # Add samples to buffer
            if use_buffer:
                assert buffer is not None and buffer_state is not None
                buffer_state = buffer.add(
                    buffer_state,
                    trajectories[:, -1],
                    log_pbs_over_pfs,
                    log_rewards,
                    subtb_losses.sum(-1),
                )

        # Off-policy training with buffer samples
        else:
            assert buffer is not None and buffer_state is not None

            # Sample terminal states from buffer
            key, key_gen = jax.random.split(key_gen)
            samples, indices = buffer.sample(buffer_state, key, alg_cfg.batch_size)

            # Get grads with the off-policy samples
            grads, (
                _,  # trajectories
                log_pbs_over_pfs,
                _,  # subtb_discrepancy
                log_rewards,
                subtb_losses,
            ) = loss_bwd_grad_fn(key, model_state, model_state.params, samples)
            model_state = model_state.apply_gradients(grads=grads)

            # Update scores in buffer if needed
            if alg_cfg.buffer.update_score:
                buffer_state = buffer.update_priority(
                    buffer_state, indices, log_pbs_over_pfs, log_rewards, subtb_losses.sum(-1)
                )

        if cfg.use_wandb:
            wandb.log({"loss": jnp.mean(subtb_losses)}, step=it)

        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * alg_cfg.batch_size)  # FIXME

            logger.update(eval_fn(model_state, key))

            # Evaluate buffer samples
            if use_buffer:
                assert buffer is not None and buffer_state is not None
                from eval import discrepancies

                buffer_xs, _ = buffer.sample(buffer_state, key, cfg.eval_samples)

                for d in cfg.discrepancies:
                    if logger.get(f"discrepancies/buffer_samples_{d}", None) is None:
                        logger[f"discrepancies/buffer_samples_{d}"] = []
                    logger[f"discrepancies/buffer_samples_{d}"].append(
                        getattr(discrepancies, f"compute_{d}")(target_xs, buffer_xs, cfg)
                        if target_xs is not None
                        else jnp.inf
                    )
                viz_dict = target.visualise(samples=buffer_xs, show=cfg.visualize_samples).items()
                logger.update({f"{key}_buffer_samples": value for key, value in viz_dict})

            print_results(it, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger), step=it)
