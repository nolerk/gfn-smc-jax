import os
from datetime import datetime

import comet_ml
import wandb
import hydra
import jax
import matplotlib
from omegaconf import DictConfig, OmegaConf

from utils.helper import flatten_dict, reset_device_memory
from utils.logger import init_logger, log_code, set_summary, finish
from utils.train_selector import get_train_fn


@hydra.main(version_base=None, config_path="configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Load the chosen algorithm-specific configuration dynamically
    cfg = hydra.utils.instantiate(cfg)
    target = cfg.target.fn

    # Determine run name
    run_name = cfg.wandb.get("name")
    if not run_name:
        run_name = (
            f"{cfg.algorithm.name}_{cfg.target.name}_{target.dim}_{datetime.now()}_seed{cfg.seed}"
        )
        cfg.wandb.name = run_name

    if not cfg.visualize_samples:
        matplotlib.use("agg")

    # Initialize logger if enabled
    if cfg.use_logger:
        logger = init_logger(
            cfg,
            config_dict=flatten_dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        )
        log_code(".")
    
    train_fn = get_train_fn(cfg.algorithm.name)

    try:
        if cfg.use_jit:
            train_fn(cfg, target)
        else:
            with jax.disable_jit():
                train_fn(cfg, target)
        if cfg.use_logger:
            set_summary("error", None)
            finish()

    except Exception as e:
        if cfg.use_logger:
            set_summary("error", str(e))
            finish(exit_code=1)
        reset_device_memory()
        raise e


if __name__ == "__main__":
    main()
