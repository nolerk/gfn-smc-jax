import jax
import jax.numpy as jnp
from jax._src.scipy.special import logsumexp

from eval import discrepancies
from eval.utils import (
    avg_stddiv_across_marginals,
    compute_reverse_ess,
    moving_averages,
    save_samples,
)


def get_eval_fn(cfg, target, target_samples):
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
        "ESS/forward": [],
        "ESS/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "discrepancies/mmd_target": [],
        "discrepancies/sd_target": [],
        "other/target_log_prob": [],
        "other/delta_mean_marginal_std": [],
        "other/EMC": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    n_eval_samples = cfg.eval_samples

    def eval_fn(samples, log_ratio, target_log_prob, fwd_log_ratio):
        ln_z = logsumexp(log_ratio) - jnp.log(n_eval_samples)
        elbo = jnp.mean(log_ratio)

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
        logger["ESS/reverse"].append(compute_reverse_ess(log_ratio, cfg.eval_samples))
        logger["other/delta_mean_marginal_std"].append(
            jnp.abs(avg_stddiv_across_marginals(samples) - target.marginal_std)
        )
        logger["other/target_log_prob"].append(jnp.mean(target_log_prob))

        if cfg.compute_forward_metrics and (target_samples is not None):
            eubo = jnp.mean(fwd_log_ratio)
            fwd_ln_z = -(jax.scipy.special.logsumexp(-fwd_log_ratio) - jnp.log(cfg.eval_samples))
            fwd_ess = jnp.exp(
                fwd_ln_z - (jax.scipy.special.logsumexp(fwd_log_ratio) - jnp.log(cfg.eval_samples))
            )
            if target.log_Z is not None:
                Z_eubo = jnp.exp(eubo)
                logger["Z/delta_eubo"].append(jnp.abs(Z_eubo - Z_ground_truth))

            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_ln_z - target.log_Z))
                Z_forward = jnp.exp(fwd_ln_z)
                logger["Z/delta_forward"].append(jnp.abs(Z_forward - Z_ground_truth))
            logger["logZ/forward"].append(fwd_ln_z)
            logger["KL/eubo"].append(eubo)
            logger["ESS/forward"].append(fwd_ess)

        logger.update(target.visualise(samples=samples))

        if cfg.compute_emc and cfg.target.has_entropy:
            logger["other/EMC"].append(target.entropy(samples))

        for d in cfg.discrepancies:
            logger[f"discrepancies/{d}"].append(
                getattr(discrepancies, f"compute_{d}")(target_samples, samples, cfg)
                if target_samples is not None
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
                            target_samples, samples_to_compare, cfg
                        )
                        if target_samples is not None
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
                    logger[f"model_selection/{key}_MAX"].append(max(logger[key]))
                    logger[f"model_selection/{key}_MIN"].append(min(logger[key]))
                else:
                    logger[key] = [value]
                    logger[f"model_selection/{key}_MAX"] = [max(logger[key])]
                    logger[f"model_selection/{key}_MIN"] = [min(logger[key])]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return eval_fn, logger
