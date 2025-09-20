import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map

from algorithms.common.models.pisgrad_net import PISGRADNet


def init_model(key, dim, alg_cfg) -> TrainState:
    def build_lr_schedule(base_lr):
        sched_cfg = getattr(alg_cfg, "lr_schedule", None)
        if sched_cfg is None:
            return lambda step: base_lr

        sched_type = getattr(sched_cfg, "type", "constant")
        match sched_type:
            case "constant":
                return lambda step: base_lr
            case "multistep":
                milestones_arr = (
                    jnp.array(sched_cfg.milestones, dtype=jnp.int32)
                    if len(sched_cfg.milestones) > 0
                    else None
                )

                def multistep_fn(step):
                    if milestones_arr is None:
                        num_decays = 0
                    else:
                        num_decays = jnp.sum(step >= milestones_arr)
                    return base_lr * (sched_cfg.gamma**num_decays)

                return multistep_fn
            case "cosine":
                decay_steps = max(alg_cfg.iters, 1)
                return optax.cosine_decay_schedule(
                    init_value=base_lr, decay_steps=decay_steps, alpha=sched_cfg.end_factor
                )
            case _:
                raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

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

    if "gfn" not in alg_cfg.name:
        optimizer = optax.chain(
            optax.zero_nans(),
            (
                optax.clip_by_global_norm(alg_cfg.grad_clip)
                if alg_cfg.grad_clip > 0
                else optax.identity()
            ),
            optax.adam(learning_rate=build_lr_schedule(alg_cfg.step_size)),
        )
    else:
        if alg_cfg.name == "gfn_tb":
            optimizers_map = {
                "network_optim": optax.adam(learning_rate=build_lr_schedule(alg_cfg.step_size))
            }
            if alg_cfg.loss_type == "tb":
                additional_params = {"logZ": jnp.array((alg_cfg.init_logZ,))}
                params["params"] = {**params["params"], **additional_params}
                optimizers_map["logZ_optim"] = optax.adam(
                    learning_rate=build_lr_schedule(alg_cfg.logZ_step_size)
                )
            param_labels = path_aware_map(
                lambda path, _: "logZ_optim" if "logZ" in path else "network_optim", params
            )
        elif alg_cfg.name == "gfn_subtb" or alg_cfg.name == "gfn_tbsubtb":
            optimizers_map = {
                "network_optim": optax.adam(learning_rate=build_lr_schedule(alg_cfg.step_size)),
                "logflow_optim": optax.adam(
                    learning_rate=build_lr_schedule(alg_cfg.logflow_step_size)
                ),
            }
            if alg_cfg.reference_process in ["ou", "ou_dds"]:
                additional_params = {"logZ": jnp.array((alg_cfg.init_logZ,))}
                params["params"] = {**params["params"], **additional_params}
                optimizers_map["logZ_optim"] = optax.adam(
                    learning_rate=build_lr_schedule(alg_cfg.logZ_step_size)
                )

            def label_map(path, _):
                if (
                    "flow_state_time_net" in path
                    or "flow_time_coder_state" in path
                    or "flow_timestep_phase" in path
                ):
                    return "logflow_optim"
                elif "logZ" in path:
                    return "logZ_optim"
                else:
                    return "network_optim"

            param_labels = path_aware_map(label_map, params)

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

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state
