import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map

from algorithms.common.models.pisgrad_net import PISGRADNet


def init_model(key, dim, alg_cfg) -> TrainState:
    # Define the model
    model = PISGRADNet(**alg_cfg.model)
    # model = LangevinNetwork(**alg_cfg.model)
    key, key_gen = jax.random.split(key)
    params = model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )

    if alg_cfg.name == "gfn_tb" and alg_cfg.loss_type == "tb":
        additional_params = {"logZ": jnp.array((alg_cfg.init_logZ,))}
        params["params"] = {**params["params"], **additional_params}

        optimizers_map = {
            "network_optim": optax.adam(learning_rate=alg_cfg.step_size),
            "logZ_optim": optax.adam(learning_rate=alg_cfg.logZ_step_size),
        }
        param_labels = path_aware_map(
            lambda path, _: "logZ_optim" if "logZ" in path else "network_optim", params
        )
        partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)
        optimizer = optax.chain(
            optax.zero_nans(),
            (
                optax.clip_by_global_norm(alg_cfg.grad_clip)
                if alg_cfg.grad_clip > 0
                else optax.identity()
            ),
            partitioned_optimizer,
        )

    else:
        optimizer = optax.chain(
            optax.zero_nans(),
            (
                optax.clip_by_global_norm(alg_cfg.grad_clip)
                if alg_cfg.grad_clip > 0
                else optax.identity()
            ),
            optax.adam(learning_rate=alg_cfg.step_size),
        )

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state
