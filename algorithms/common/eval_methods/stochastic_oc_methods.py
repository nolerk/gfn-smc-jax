from functools import partial

import jax
import jax.numpy as jnp

from eval import discrepancies
from eval.utils import (
    moving_averages,
    save_samples,
)


def get_eval_fn(rnd, target, target_xs, cfg):
    rnd_reverse = jax.jit(partial(rnd, prior_to_target=True))

    if cfg.compute_forward_metrics and target.can_sample:
        rnd_forward = jax.jit(
            partial(rnd, prior_to_target=False, terminal_xs=target_xs)
        )

    logger = {
        "KL/elbo": [],
        "KL/eubo": [],
        "logZ/delta_forward": [],
        "logZ/forward": [],
        "logZ/delta_reverse": [],
        "logZ/reverse": [],
        "Z/delta_forward": [],
        "Z/delta_reverse": [],
        "Z/delta_elbo": [],
        "Z/delta_eubo": [],
        "Z_var/reverse": [],
        "discrepancies/sd": [],
        "discrepancies/sd_target": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def short_eval(model_state, key):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        samples, running_costs, stochastic_costs, terminal_costs = rnd_reverse(
            key, model_state, *params
        )[:4]

        log_is_weights = -(running_costs + stochastic_costs + terminal_costs)
        ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)

        Z_var_reverse = jnp.var(jnp.exp(log_is_weights))
        logger["Z_var/reverse"].append(Z_var_reverse)

        elbo = jnp.mean(log_is_weights)

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(ln_z - target.log_Z))
            Z_reverse = jnp.exp(ln_z)
            Z_ground_truth = jnp.exp(target.log_Z)
            logger["Z/delta_reverse"].append(jnp.abs(Z_reverse - Z_ground_truth))

        logger["logZ/reverse"].append(ln_z)
        logger["KL/elbo"].append(elbo)

        if target.log_Z is not None:
            Z_elbo = jnp.exp(elbo)
            logger["Z/delta_elbo"].append(jnp.abs(Z_elbo - Z_ground_truth))

        if cfg.compute_forward_metrics and target.can_sample:
            (
                fwd_samples,
                fwd_running_costs,
                fwd_stochastic_costs,
                fwd_terminal_costs,
            ) = rnd_forward(jax.random.PRNGKey(0), model_state, *params)[:4]
            fwd_log_is_weights = -(
                fwd_running_costs + fwd_stochastic_costs + fwd_terminal_costs
            )
            fwd_ln_z = jax.scipy.special.logsumexp(fwd_log_is_weights) - jnp.log(
                cfg.eval_samples
            )
            eubo = jnp.mean(fwd_log_is_weights)
            if target.log_Z is not None:
                Z_eubo = jnp.exp(eubo)
                logger["Z/delta_eubo"].append(jnp.abs(Z_eubo - Z_ground_truth))

            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_ln_z - target.log_Z))
                Z_forward = jnp.exp(fwd_ln_z)
                logger["Z/delta_forward"].append(jnp.abs(Z_forward - Z_ground_truth))
            logger["logZ/forward"].append(fwd_ln_z)
            logger["KL/eubo"].append(eubo)

        logger.update(target.visualise(samples=samples))

        for d in cfg.discrepancies:
            logger[f"discrepancies/{d}"].append(
                getattr(discrepancies, f"compute_{d}")(target_xs, samples, cfg)
                if target_xs is not None
                else jnp.inf
            )
            if len(logger[f"discrepancies/{d}_target"]) == 0:
                discrepancies_target = []
                for seed in range(1, 6):
                    samples_to_compare = target.sample(
                        jax.random.PRNGKey(seed), (cfg.eval_samples,)
                    )
                    discrepancies_target.append(
                        getattr(discrepancies, f"compute_{d}")(
                            target_xs, samples_to_compare, cfg
                        )
                        if target_xs is not None
                        else jnp.inf
                    )
                logger[f"discrepancies/{d}_target"].append(
                    jnp.mean(jnp.array(discrepancies_target))
                )

        if cfg.moving_average.use_ma:
            for key, value in moving_averages(
                logger, window_size=cfg.moving_average.window_size
            ).items():
                if isinstance(value, list):
                    value = value[0]
                if key in logger.keys():
                    logger[key].append(value)
                else:
                    logger[key] = [value]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return short_eval, logger
