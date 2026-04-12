import chex
import jax
import jax.numpy as jnp
import jax.random as random


def rejection_sample_domain(
    seed: chex.PRNGKey,
    sample_shape: chex.Shape,
    sample_fn,
    is_inside_fn,
    dim: int,
) -> chex.Array:
    num_samples = sample_shape[0]

    def cond_fn(state):
        _, num_filled, *_ = state
        return num_filled < num_samples

    def body_fn(state):
        seed, num_filled, buffer = state
        seed, subkey = random.split(seed)

        N = buffer.shape[0]

        proposal = sample_fn(seed=subkey, sample_shape=(N,))
        mask = is_inside_fn(proposal)
        mask_int = mask.astype(jnp.int32)

        positions = jnp.cumsum(mask_int) - 1
        num_accept = jnp.sum(mask_int)

        remaining = N - num_filled
        num_take = jnp.minimum(num_accept, remaining)

        valid = mask & (positions < num_take)

        # replace invalid entries with dummy (won’t be used)
        safe_positions = jnp.where(valid, positions, 0)
        safe_values = jnp.where(valid[:, None], proposal, 0.0)

        target_idx = num_filled + safe_positions

        buffer = buffer.at[target_idx].add(safe_values)

        num_filled = num_filled + num_take

        return seed, num_filled, buffer

    buffer = jnp.zeros((num_samples, dim))
    init_state = (seed, 0, buffer)
    _, _, buffer = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return buffer.reshape(sample_shape + (dim,))
