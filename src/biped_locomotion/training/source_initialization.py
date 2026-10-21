# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any, Callable, Mapping, Sequence
import jax
import jax.numpy as jp

PINNED_INDICES = (49, 50, 51, 57, 58, 59, 60)


def bypass_pinned(
    normalized: jax.Array, raw: jax.Array, pinned: Sequence[int] = PINNED_INDICES
) -> jax.Array:
    """Restore raw values at ``pinned`` indices of the last axis.

    Works for any leading batch shape, including the unbatched ``(N,)`` case.
    """
    idx = jp.asarray(tuple(pinned), dtype=jp.int32)
    return normalized.at[..., idx].set(raw[..., idx])


def masked_preprocessor(
    base_fn: Callable[..., Any], pinned: Sequence[int] = PINNED_INDICES
) -> Callable[..., Any]:
    """Wrap a brax ``preprocess_observations_fn`` with the pinned bypass."""
    pinned = tuple((int(i) for i in pinned))

    def fn(obs, params):
        z = base_fn(obs, params)
        if isinstance(obs, Mapping):
            return {k: bypass_pinned(z[k], obs[k], pinned) for k in obs}
        return bypass_pinned(z, obs, pinned)

    return fn


def u13_softplus_inverse(value: float) -> float:
    """Exact inverse of ``jax.nn.softplus``, for the brax scale head.

    VERIFIED against ``brax.training.distribution.NormalTanhDistribution.
    create_dist`` (brax 0.14.2), which computes

        loc, scale_raw = split(params, 2)
        scale = (softplus(scale_raw) + min_std) * var_scale

    with the defaults ``min_std=0.001``, ``var_scale=1`` that
    ``ppo_networks.make_ppo_networks`` uses. The transform is SOFTPLUS, not
    exp, so ``bias = log(std)`` would be wrong.
    """
    import math

    return float(math.log(math.expm1(value)))


def u13_scale_bias_for_target_std(target_stds, min_std: float):
    """Returns the distribution-layer bias rows realizing ``target_stds``."""
    return jp.asarray(
        [u13_softplus_inverse(float(s) - min_std) for s in target_stds],
        dtype=jp.float32,
    )


def u13_fresh_actor_params(
    network, action_size: int, target_stds, min_std: float, seed: int
):
    """Freshly initializes the actor, then applies U13's two head edits.

    No checkpoint is read. Lives here (not in the trainer) so it carries no
    absl-flag dependency and can be exercised directly by the U13 contract
    test.

      * action-mean kernel AND bias -> EXACTLY zero, so ``tanh(mean) == 0`` for
        every observation and ``motor_target == q_ref`` on the first tick;
      * scale-head kernel -> zero and its bias -> ``softplus_inverse(std -
        min_std)``, so the initial per-actuator action std is exactly
        ``target_stds`` and is state-INDEPENDENT.

    Returns:
      ``(policy_params, distribution_layer_name, scale_bias)``.
    """
    policy = network.policy_network.init(jax.random.PRNGKey(seed))
    params = dict(policy["params"])
    output_name = max(
        (k for k in params if k.startswith("hidden_")),
        key=lambda k: int(k.split("_")[1]),
    )
    output = dict(params[output_name])
    kernel = jp.asarray(output["kernel"])
    bias = jp.asarray(output["bias"])
    if kernel.shape[-1] != 2 * action_size:
        raise ValueError(
            f"U13 expects a {2 * action_size}-wide distribution head ({action_size} means + {action_size} scales), got {kernel.shape}"
        )
    scale_bias = u13_scale_bias_for_target_std(target_stds, min_std)
    output["kernel"] = jp.zeros_like(kernel)
    output["bias"] = jp.zeros_like(bias).at[action_size:].set(scale_bias)
    params[output_name] = output
    policy = dict(policy)
    policy["params"] = params
    return (policy, output_name, scale_bias)
