import jax
import jax.numpy as jnp

from eval import discrepancies
from eval.utils import moving_averages, save_samples


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
        "discrepancies/sd": [],
        "discrepancies/sd_target": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def eval_fn(samples, elbo, rev_lnz, eubo, fwd_lnz):

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(rev_lnz - target.log_Z))
            Z_reverse = jnp.exp(rev_lnz)
            Z_ground_truth = jnp.exp(target.log_Z)
            logger["Z/delta_reverse"].append(jnp.abs(Z_reverse - Z_ground_truth))

        logger["logZ/reverse"].append(rev_lnz)
        logger["KL/elbo"].append(elbo)

        if target.log_Z is not None:
            Z_elbo = jnp.exp(elbo)
            logger["Z/delta_elbo"].append(jnp.abs(Z_elbo - Z_ground_truth))

        if (
            cfg.compute_forward_metrics
            and (target_samples is not None)
            and (fwd_lnz is not None)
        ):
            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_lnz - target.log_Z))
                Z_forward = jnp.exp(fwd_lnz)
                logger["Z/delta_forward"].append(jnp.abs(Z_forward - Z_ground_truth))
                Z_eubo = jnp.exp(eubo)
                logger["Z/delta_eubo"].append(jnp.abs(Z_eubo - Z_ground_truth))
            logger["logZ/forward"].append(fwd_lnz)
            logger["KL/eubo"].append(eubo)

        logger.update(target.visualise(samples=samples))

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
                else:
                    logger[key] = [value]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return eval_fn, logger
