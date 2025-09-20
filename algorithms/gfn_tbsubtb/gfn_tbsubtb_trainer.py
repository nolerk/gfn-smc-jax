"""
Code for alternate training of Trajectory Balance (TB) and Sub-Trajectory Balance (SubTB).
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.gfn_subtb.buffer import build_intermediate_state_buffer
from algorithms.gfn_subtb.gfn_subtb_rnd import rnd
from algorithms.gfn_subtb.visualise import (
    visualise_intermediate_distribution,
    visualise_true_intermediate_distribution,
)
from algorithms.gfn_tbsubtb.gfn_tbsubtb_rnd import loss_fn_joint, loss_fn_tb, loss_fn_subtb
from algorithms.gfn_tb.utils import get_invtemp
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def gfn_tbsubtb_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size
    num_steps = alg_cfg.num_steps
    n_chunks = alg_cfg.n_chunks
    reference_process = alg_cfg.reference_process
    noise_schedule = alg_cfg.noise_schedule

    # Define initial and target density
    if reference_process == "pinned_brownian":  # Following PIS
        initial_dist = None  # actually, the initial distribution is Dirac delta at the origin
        aux_tuple = (dim,)
    elif reference_process in ["ou", "ou_dds"]:  # DIS or DDS
        initial_dist = distrax.MultivariateNormalDiag(
            jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
        )
        aux_tuple = (alg_cfg.init_std, initial_dist.sample, initial_dist.log_prob)
        if reference_process == "ou_dds":
            aux_tuple = (*aux_tuple, alg_cfg.noise_scale)  # DDS
    else:
        raise ValueError(f"Reference process {reference_process} not supported.")

    # Initialize the buffer
    use_buffer = alg_cfg.buffer.use
    buffer = buffer_state = buffer_cfg = None
    if use_buffer:
        buffer_cfg = alg_cfg.buffer
        buffer = build_intermediate_state_buffer(
            dim=dim,
            max_length=buffer_cfg.max_length_in_batches * batch_size,
            prioritize_by=buffer_cfg.prioritize_by,
            target_ess=buffer_cfg.target_ess,
            sampling_method=buffer_cfg.sampling_method,
            rank_k=buffer_cfg.rank_k,
        )
        buffer_state = buffer.init(dtype=target_xs.dtype, device=target_xs.device)

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    rnd_partial_base = partial(
        rnd,
        reference_process=reference_process,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=num_steps,
        noise_schedule=noise_schedule,
        use_lp=alg_cfg.model.use_lp,
        partial_energy=alg_cfg.partial_energy,
    )
    if alg_cfg.alternate:
        loss_fn_base = partial(loss_fn_tb, loss_type=alg_cfg.loss_type)
    else:
        loss_fn_base = partial(
            loss_fn_joint,
            loss_type=alg_cfg.loss_type,
            n_chunks=n_chunks,
            subtb_weight=alg_cfg.subtb_weight,
        )

    # Define the function to be JIT-ed for FWD pass
    @jax.jit
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params, invtemp=1.0):
        rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_partial=rnd_p, invtemp=invtemp)

    # Define the function to be JIT-ed for FWD pass without gradients
    @jax.jit
    def loss_fwd_nograd_fn(key, model_state, params, invtemp=1.0):
        # prior_to_target=True, terminal_xs=None
        rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_partial=rnd_p, invtemp=invtemp)

    # Define the function to be JIT-ed for BWD pass
    @jax.jit
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_bwd_grad_fn(key, model_state, params, terminal_xs, log_rewards, invtemp=1.0):
        # prior_to_target=False, terminal_xs is now an argument
        rnd_p = partial(
            rnd_partial_base,
            batch_size=batch_size,
            prior_to_target=False,
            terminal_xs=terminal_xs,
            log_rewards=log_rewards,
        )
        return loss_fn_base(key, model_state, params, rnd_partial=rnd_p, invtemp=invtemp)

    loss_bwd_grad_fn_flow = lambda *args, **kwargs: None
    if alg_cfg.alternate:
        loss_fn_flow_base = partial(loss_fn_subtb, n_chunks=n_chunks)

        @jax.jit
        @partial(jax.grad, argnums=2, has_aux=True)
        def loss_bwd_grad_fn_flow(key, model_state, params, terminal_xs, log_rewards, invtemp=1.0):
            rnd_p = partial(
                rnd_partial_base,
                batch_size=batch_size,
                prior_to_target=False,
                terminal_xs=terminal_xs,
                log_rewards=log_rewards,
            )
            return loss_fn_flow_base(key, model_state, params, rnd_partial=rnd_p, invtemp=invtemp)

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(rnd_partial_base, batch_size=cfg.eval_samples),
        target,
        target_xs,
        cfg,
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    ### Plot the True intermediate distributions
    if cfg.use_wandb and getattr(target, "log_prob_t", None) is not None:
        true_vis_dict = visualise_true_intermediate_distribution(
            target.visualise,
            [i * (num_steps // n_chunks) for i in range(n_chunks + 1)],
            num_steps,
            reference_process,
            noise_schedule,
            alg_cfg.init_std,
            alg_cfg.noise_scale,
            target.log_prob,
            target.log_prob_t,
        )
        wandb.log(
            {
                key.replace("figures/", "figures_true/"): value
                for key, value in true_vis_dict.items()
            },
            step=0,
        )

    ### Prefill phase
    if use_buffer and buffer_cfg.prefill_steps > 0:
        # Define the function to be JIT-ed for FWD pass
        assert buffer is not None and buffer_state is not None
        for _ in range(buffer_cfg.prefill_steps):
            key, key_gen = jax.random.split(key_gen)
            _, (
                trajectories,
                log_pbs_over_pfs,
                log_rewards,
                tb_losses,
                subtb_losses,
            ) = loss_fwd_nograd_fn(key, model_state, model_state.params)
            buffer_state = buffer.add(
                buffer_state,
                trajectories[:, -1],
                log_pbs_over_pfs.sum(-1),
                log_rewards,
                tb_losses,
            )

    ### Training phase
    for it in range(alg_cfg.iters):
        invtemp = get_invtemp(
            it, alg_cfg.iters // 2, alg_cfg.init_invtemp, (alg_cfg.init_invtemp < 1.0)
        )

        # On-policy training with forward samples
        if not use_buffer or it % (buffer_cfg.bwd_to_fwd_ratio + 1) == 0:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (trajectories, log_pbs_over_pfs, log_rewards, tb_losses, subtb_losses) = (
                loss_fwd_grad_fn(key, model_state, model_state.params, invtemp=invtemp)
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Add samples to buffer
            if use_buffer:
                assert buffer is not None and buffer_state is not None
                buffer_state = buffer.add(
                    buffer_state,
                    trajectories[:, -1],
                    log_pbs_over_pfs.sum(-1),
                    log_rewards,
                    tb_losses,
                )

        # Off-policy training with buffer samples
        else:
            assert buffer is not None and buffer_state is not None

            # Sample terminal states from buffer
            key, key_gen = jax.random.split(key_gen)
            samples, log_rewards, indices = buffer.sample(buffer_state, key, batch_size)

            # Get grads with the off-policy samples
            key, key_gen = jax.random.split(key_gen)
            grads, (trajectories, log_pbs_over_pfs, log_rewards, tb_losses, subtb_losses) = (
                loss_bwd_grad_fn(
                    key, model_state, model_state.params, samples, log_rewards, invtemp=invtemp
                )
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Update scores in buffer if needed
            if buffer_cfg.update_score:
                buffer_state = buffer.update_priority(
                    buffer_state, indices, log_pbs_over_pfs, log_rewards, tb_losses
                )

        if cfg.use_wandb:
            wandb.log(
                {
                    "tb_loss": jnp.mean(tb_losses),
                    "subtb_loss": jnp.mean(subtb_losses.sum(-1)),
                    "logZ_learned": model_state.params["params"]["logZ"],
                },
                step=it,
            )

        if alg_cfg.alternate and it > 0 and it % alg_cfg.learn_flow_every == 0:
            print(f"Learning flow at iteration {it}")
            assert buffer is not None and buffer_state is not None

            for _ in range(alg_cfg.learn_flow_iters):
                key, key_gen = jax.random.split(key_gen)
                samples, log_rewards, indices = buffer.sample(buffer_state, key, batch_size)

                key, key_gen = jax.random.split(key_gen)
                grads, (
                    trajectories,
                    log_pbs_over_pfs,
                    log_rewards,
                    _,
                    subtb_losses,
                ) = loss_bwd_grad_fn_flow(
                    key, model_state, model_state.params, samples, log_rewards, invtemp=invtemp
                )
                model_state = model_state.apply_gradients_flow(grads=grads)

            if cfg.use_wandb:
                wandb.log({f"alt_subtb_loss": jnp.mean(subtb_losses.sum(-1))}, step=it)

        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * batch_size)  # FIXME

            logger.update(eval_fn(model_state, key))

            # Visualize intermediate distributions (learned flows)
            # Obtain trajectories from the model
            if cfg.use_wandb:
                key, key_gen = jax.random.split(key_gen)
                vis_dict = visualise_intermediate_distribution(
                    target.visualise,
                    [i * (num_steps // n_chunks) for i in range(n_chunks)],
                    num_steps,
                    None,  # trajectories,
                    model_state,
                    alg_cfg.partial_energy,
                    batch_size,
                    reference_process,
                    noise_schedule,
                    initial_dist,
                    alg_cfg.noise_scale,
                    target.log_prob,
                )
                logger.update(vis_dict)

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
                if cfg.use_wandb:
                    vis_dict = target.visualise(samples=buffer_xs)
                    logger.update(
                        {f"{key}_buffer_samples": value for key, value in vis_dict.items()}
                    )

            print_results(it, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger), step=it)
