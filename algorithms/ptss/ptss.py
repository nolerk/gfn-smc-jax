from typing import Callable
from collections import UserDict

import jax
import jax.numpy as jnp


class StateDict(UserDict):
    def __init__(self, dict_: dict = None, /, **kwargs):
        super().__init__()
        if dict_ is not None:
            self.data.update(dict_)

        if kwargs:
            self.data.update(kwargs)
        
    def __getattr__(self, name: str):
        if name not in self.data:
            raise AttributeError(f"'StateDict' object has no attribute '{name}'")
        return self.data[name]
    
    def unpack(self, *keys):
        return (self.data[key] for key in keys)


#  Register StateDict as a pytree for traceablility in JAX, 
#  preserving permutation invariance
def flatten_state_dict(state_dict):
    # Sort keys to guarantee canonical order
    keys = sorted(state_dict.keys())
    
    # Extract values in the specific order of the sorted keys
    values = [state_dict[k] for k in keys]
    
    # Return values (children) and keys (auxiliary data)
    # Note: Use tuple(keys) because aux data should be hashable
    return values, tuple(keys)

def unflatten_state_dict(keys, values):
    return StateDict(zip(keys, values))

jax.tree_util.register_pytree_node(
    StateDict,
    flatten_state_dict,
    unflatten_state_dict
)


def ptss_swap(
        x: jnp.ndarray, 
        log_density: Callable, 
        temps: jnp.ndarray, 
        key: jax.random.PRNGKey
    ):
    n_temps, batch, *_,  = x.shape
    key, k1, k2 = jax.random.split(key, 3)    
    
    #  1. Select indices
    tau_idx1 = jax.random.randint(k1, (batch,), 0, n_temps - 1)
    tau_idx2 = tau_idx1 + 1

    #  2. Compute acceptance probability 
    batch_idx = jnp.arange(batch)
    x1 = x[tau_idx1, batch_idx]
    x2 = x[tau_idx2, batch_idx]

    # dE = E1 - E2 = logp(x2) - logp(x1)
    log_p_1 = log_density(x1)
    log_p_2 = log_density(x2)
    dE = log_p_2 - log_p_1

    beta_1 = 1.0 / temps[tau_idx1]
    beta_2 = 1.0 / temps[tau_idx2]
    log_A = dE * (beta_1 - beta_2)
    
    log_U = jnp.log(jax.random.uniform(k2, (batch,)))
    swap_mask = log_U < log_A

    # 3. Swap states
    new_values_for_tau1 = jnp.where(swap_mask[:, None], x2, x1)
    new_values_for_tau2 = jnp.where(swap_mask[:, None], x1, x2)

    x = x.at[tau_idx1, batch_idx].set(new_values_for_tau1)
    x = x.at[tau_idx2, batch_idx].set(new_values_for_tau2)

    return StateDict(x=x, key=key, swap_avg=swap_mask.mean())


def sample_new_x(
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        log_density: Callable, 
        window_size: float, 
        key: jax.random.PRNGKey, 
        max_while_iter: int = 1_000,
        limits: tuple = (-5, 5)
    ):
    lim_left, lim_right = limits
    left, right = x - window_size / 2, x + window_size / 2

    def _expand_body(state):
        val, sign = state.val, state.sign
        val = jnp.where(state.mask, val + sign * window_size, val)
        
        density_cond = (log_density(val) >= jnp.log(y))[..., None]
        border_cond = jnp.where(sign == -1, val >= lim_left, val <= lim_right)
        mask = density_cond & border_cond

        return StateDict(val=val, mask=mask, sign=sign, iter=state.iter + 1)
    
    def _cond(state):
        return jnp.any(state.mask) & (state.iter < max_while_iter)
    
    left_upd_mask = (log_density(left) >= jnp.log(y))[..., None] & (left >= lim_left)
    left_state = StateDict(val=left, mask=left_upd_mask, sign=-1, iter=0)  
    left = jax.lax.while_loop(_cond, _expand_body, left_state).val

    right_upd_mask = \
        (log_density(right) >= jnp.log(y))[..., None] & (right <= lim_right)
    right_state = StateDict(val=right, mask=right_upd_mask, sign=1, iter=0)
    right = jax.lax.while_loop(_cond, _expand_body, right_state).val

    key, subkey = jax.random.split(key)
    x_new = jax.random.uniform(subkey, x.shape) * (right - left) + left
    mask = (log_density(x_new) < jnp.log(y))[..., None]
    
    def _sample_body(state):
        x, x_new, left, right, mask = \
            state.unpack("x", "x_new", "left", "right", "mask")
        
        right = jnp.where(mask & (x_new > x), x_new, right)
        left = jnp.where(mask & (x_new < x), x_new, left)

        key, subkey = jax.random.split(state.key)
        x_new = jax.random.uniform(subkey, x.shape) * (right - left) + left
        mask = (log_density(x_new) < jnp.log(y))[..., None]

        new_state = StateDict(
            x=x, x_new=x_new, left=left, right=right, 
            mask=mask, key=key, iter=state.iter + 1
        )
        
        return new_state

    sample_state = StateDict(
        x=x, x_new=x_new, left=left, 
        right=right, mask=mask, key=key, iter=0
    )
    
    x_new = jax.lax.while_loop(_cond, _sample_body, sample_state).x_new
    
    return x_new


def parallel_tempering_with_slice_sampling(
        x_start: jnp.ndarray, 
        log_density: Callable,
        temps: jnp.ndarray, 
        window_size: float = 0.5, 
        chain_length: int = 1_000, 
        swap_freq: int = 10,
        alpha: float = 1.0, 
        seed: int = 0,
        max_while_iter: int = 1_000,
        propose_region_limits: tuple = (-5, 5)
    ):
    key = jax.random.PRNGKey(seed)
    
    tempered_log_density = lambda x: jax.vmap(log_density)(x) / temps[:, None]

    def _ptss_one_step(carry, iteration):

        key, subkey = jax.random.split(carry.key)
        gamma_sample = jax.random.gamma(subkey, alpha, shape=1)
        y = jnp.exp(tempered_log_density(carry.x) -gamma_sample)

        new_x = sample_new_x(
            carry.x, y, tempered_log_density, window_size, key, 
            max_while_iter=max_while_iter,
            limits=propose_region_limits
        )
        
        swap_output = jax.lax.cond(
            iteration % swap_freq == 0,
            lambda x: ptss_swap(x, log_density, temps, key),
            lambda x: StateDict(x=new_x, key=key, swap_avg=0.0),
            new_x
        )

        jax.lax.cond(
            iteration % swap_freq == 0,
            lambda: jax.debug.print(
                "Progress: step {i}, swap avg: {swap:.3f}", 
                i=iteration, swap=swap_output.swap_avg
            ),
            lambda: None
        )
        
        return StateDict(x=swap_output.x, key=swap_output.key), swap_output.x
    
    x = jnp.tile(jnp.expand_dims(x_start, axis=0), (temps.shape[0], 1, 1))
    carry = StateDict(x=x, key=key)
    _, chain_states = jax.lax.scan(
        _ptss_one_step, carry, jnp.arange(1, chain_length + 1)
    )
    chain_states = jnp.concatenate([x[None, ...], chain_states], axis=0)

    return chain_states
