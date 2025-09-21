import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map
from flax import struct
from typing import Any

from algorithms.common.models.pisgrad_net import PISGRADNet


def pisgrad_net_label_map(path, _):
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
    elif "gfn" in alg_cfg.name:
        optimizers_map = {
            "network_optim": optax.adam(learning_rate=build_lr_schedule(alg_cfg.step_size)),
            "logflow_optim": optax.adam(learning_rate=build_lr_schedule(alg_cfg.logflow_step_size)),
        }
        if (alg_cfg.name == "gfn_tb") or (alg_cfg.reference_process in ["ou", "ou_dds"]):
            additional_params = {"logZ": jnp.array((alg_cfg.init_logZ,))}
            params["params"] = {**params["params"], **additional_params}
            optimizers_map["logZ_optim"] = optax.adam(
                learning_rate=build_lr_schedule(alg_cfg.logZ_step_size)
            )

        param_labels = path_aware_map(pisgrad_net_label_map, params)
        partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)
        if alg_cfg.name == "gfn_tb" or ("alt" not in alg_cfg.loss_type):
            optimizer = optax.chain(
                optax.zero_nans(),
                (
                    optax.clip_by_global_norm(alg_cfg.grad_clip)
                    if alg_cfg.grad_clip > 0
                    else optax.identity()
                ),
                partitioned_optimizer,
            )
        else:  # gfn_subtb and alternate (loss_type in ["tb_subtb_alt", "lv_subtb_alt"])
            # Construct boolean masks per-parameter for flow-only vs tb-only updates
            flow_mask = jax.tree_util.tree_map(lambda lbl: lbl == "logflow_optim", param_labels)
            policy_mask = jax.tree_util.tree_map(lambda lbl: lbl != "logflow_optim", param_labels)

            # TB optimizer (policy + logZ only; freeze flow params entirely)
            policy_tx = optax.chain(
                optax.zero_nans(),
                (
                    optax.clip_by_global_norm(alg_cfg.grad_clip)
                    if alg_cfg.grad_clip > 0
                    else optax.identity()
                ),
                optax.masked(partitioned_optimizer, policy_mask),
            )

            # Flow-only optimizer: mask out non-flow params so their params and opt-state don't step
            flow_tx = optax.chain(
                optax.zero_nans(),
                (
                    optax.clip_by_global_norm(alg_cfg.grad_clip)
                    if alg_cfg.grad_clip > 0
                    else optax.identity()
                ),
                optax.masked(partitioned_optimizer, flow_mask),
            )

            # Return an extended TrainState carrying both tx's and states.
            @struct.dataclass
            class AlternateTrainState(TrainState):
                flow_tx: Any = struct.field(pytree_node=False)
                flow_opt_state: Any

                def apply_gradients(self, *, grads, **kwargs):
                    # Default to TB optimizer (keeps existing call sites working)
                    updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
                    new_params = optax.apply_updates(self.params, updates)
                    return self.replace(
                        step=self.step + 1,
                        params=new_params,
                        opt_state=new_opt_state,
                    )

                def apply_gradients_flow(self, *, grads, **kwargs):
                    updates, new_flow_opt_state = self.flow_tx.update(
                        grads, self.flow_opt_state, self.params
                    )
                    new_params = optax.apply_updates(self.params, updates)
                    return self.replace(
                        step=self.step + 1,
                        params=new_params,
                        flow_opt_state=new_flow_opt_state,
                    )

            # Initialize both optimizer states
            tb_opt_state = policy_tx.init(params)
            flow_opt_state = flow_tx.init(params)

            model_state = AlternateTrainState(
                step=0,
                apply_fn=model.apply,
                params=params,
                tx=policy_tx,
                opt_state=tb_opt_state,
                flow_tx=flow_tx,
                flow_opt_state=flow_opt_state,
            )
            return model_state
    else:
        raise ValueError(f"Invalid algorithm name: {alg_cfg.name}")

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state
