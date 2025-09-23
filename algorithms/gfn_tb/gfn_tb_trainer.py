"""
Code for Trajectory Balance (TB) training.
For further details see: https://arxiv.org/abs/2301.12594 and https://arxiv.org/abs/2501.06148
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.gfn_tb.buffer import build_terminal_state_buffer
from algorithms.gfn_tb.gfn_tb_rnd import rnd, loss_fn
from algorithms.gfn_tb.utils import get_invtemp
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def gfn_tb_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define initial and target density
    if alg_cfg.reference_process == "pinned_brownian":  # Following PIS
        initial_dist = None
        aux_tuple = (dim,)
    elif alg_cfg.reference_process in ["ou", "ou_dds"]:  # DIS or DDS
        initial_dist = distrax.MultivariateNormalDiag(
            jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
        )
        aux_tuple = (alg_cfg.init_std, initial_dist.log_prob)
        if alg_cfg.reference_process == "ou_dds":
            aux_tuple = (*aux_tuple, alg_cfg.noise_scale)  # DDS
    else:
        raise ValueError(f"Reference process {alg_cfg.reference_process} not supported.")

    # Initialize the buffer
    use_buffer = alg_cfg.buffer.use
    buffer = buffer_state = None
    if use_buffer:
        buffer = build_terminal_state_buffer(
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
        initial_dist=initial_dist,
    )
    loss_fn_base = partial(loss_fn, loss_type=alg_cfg.loss_type)

    # Define the function to be JIT-ed for FWD pass
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params, invtemp=1.0):
        # prior_to_target=True, terminal_xs=None
        rnd_p = partial(rnd_partial_base, batch_size=alg_cfg.batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    # --- Define the function to be JIT-ed for BWD pass ---
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_bwd_grad_fn(key, model_state, params, terminal_xs, log_rewards, invtemp=1.0):
        # prior_to_target=False, terminal_xs is now an argument
        rnd_p = partial(
            rnd_partial_base,
            batch_size=alg_cfg.batch_size,
            prior_to_target=False,
            terminal_xs=terminal_xs,
            log_rewards=log_rewards,
        )
        return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(rnd_partial_base, batch_size=cfg.eval_samples), target, target_xs, cfg
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    ### Prefill phase
    if use_buffer and alg_cfg.buffer.prefill_steps > 0:
        # Define the function to be JIT-ed for FWD pass
        @partial(jax.jit)
        def loss_fwd_nograd_fn(key, model_state, params, invtemp=1.0):
            # prior_to_target=True, terminal_xs=None
            rnd_p = partial(rnd_partial_base, batch_size=alg_cfg.batch_size, prior_to_target=True)
            return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

        assert buffer is not None and buffer_state is not None
        for _ in range(alg_cfg.buffer.prefill_steps):
            key, key_gen = jax.random.split(key_gen)
            _, (xs, log_pbs_over_pfs, log_rewards, losses) = loss_fwd_nograd_fn(
                key, model_state, model_state.params
            )
            buffer_state = buffer.add(
                buffer_state, xs, (log_pbs_over_pfs + log_rewards), log_rewards, losses
            )

    ### Training phase
    for it in range(alg_cfg.iters):
        invtemp = get_invtemp(
            it, alg_cfg.iters // 2, alg_cfg.init_invtemp, (alg_cfg.init_invtemp < 1.0)
        )

        # On-policy training with forward samples
        if not use_buffer or it % (alg_cfg.buffer.bwd_to_fwd_ratio + 1) == 0:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (xs, log_pbs_over_pfs, log_rewards, losses) = loss_fwd_grad_fn(
                key, model_state, model_state.params, invtemp=invtemp
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Add samples to buffer
            if use_buffer:
                assert buffer is not None and buffer_state is not None
                buffer_state = buffer.add(
                    buffer_state, xs, (log_pbs_over_pfs + log_rewards), log_rewards, losses
                )

        # Off-policy training with buffer samples
        else:
            assert buffer is not None and buffer_state is not None

            # Sample terminal states from buffer
            key, key_gen = jax.random.split(key_gen)
            samples, log_rewards, indices = buffer.sample(buffer_state, key, alg_cfg.batch_size)

            # Get grads with the off-policy samples
            key, key_gen = jax.random.split(key_gen)
            grads, (_, log_pbs_over_pfs, _, losses) = loss_bwd_grad_fn(
                key, model_state, model_state.params, samples, log_rewards, invtemp=invtemp
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Update scores in buffer if needed
            if alg_cfg.buffer.update_score:
                buffer_state = buffer.update_priority(
                    buffer_state, indices, (log_pbs_over_pfs + log_rewards), log_rewards, losses
                )

        if cfg.use_wandb:
            wandb.log({"loss": jnp.mean(losses)}, step=it)
            if alg_cfg.loss_type == "tb":
                wandb.log({"logZ_learned": model_state.params["params"]["logZ"]}, step=it)

        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * alg_cfg.batch_size)  # FIXME

            logger.update(eval_fn(model_state, key))

            # Evaluate buffer samples
            if use_buffer:
                assert buffer is not None and buffer_state is not None
                from eval import discrepancies

                buffer_xs, _, _ = buffer.sample(buffer_state, key, cfg.eval_samples)

                for d in cfg.discrepancies:
                    if logger.get(f"discrepancies/buffer_samples_{d}", None) is None:
                        logger[f"discrepancies/buffer_samples_{d}"] = []
                    logger[f"discrepancies/buffer_samples_{d}"].append(
                        getattr(discrepancies, f"compute_{d}")(target_xs, buffer_xs, cfg)
                        if target_xs is not None
                        else jnp.inf
                    )
                vis_dict = target.visualise(samples=buffer_xs)
                logger.update({f"{key}_buffer_samples": value for key, value in vis_dict.items()})

            print_results(it, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger), step=it)
