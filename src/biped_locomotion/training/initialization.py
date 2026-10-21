"""Scratch provenance and exact zero-mean U13 actor head initialization."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScratchContract:
    source_checkpoint: None = None
    fresh_actor: bool = True
    fresh_critic: bool = True
    fresh_optimizer: bool = True
    fresh_normalizer: bool = True
    frozen_layers: tuple = ()

    def validate(self):
        if (
            self.source_checkpoint is not None
            or self.frozen_layers
            or not all(
                (
                    self.fresh_actor,
                    self.fresh_critic,
                    self.fresh_optimizer,
                    self.fresh_normalizer,
                )
            )
        ):
            raise ValueError("U13 must start entirely fresh")


def fresh_head(kernel, bias, target_std, min_std=0.001):
    kernel, bias, std = map(np.asarray, (kernel, bias, target_std))
    if (
        kernel.shape[-1] != 24
        or bias.shape != (24,)
        or std.shape != (12,)
        or np.any(std <= min_std)
    ):
        raise ValueError("12 means and 12 valid scale logits required")
    b = np.zeros_like(bias, dtype=float)
    b[12:] = np.log(np.expm1(std - min_std))
    return np.zeros_like(kernel), b


def initialize_u13(network, observation_size, target_stds, seed):
    from .source_initialization import u13_fresh_actor_params
    from brax.training.acme import running_statistics
    import jax.numpy as jp

    policy, _, _ = u13_fresh_actor_params(network, 12, target_stds, 0.001, seed)
    normalizer = running_statistics.init_state(
        {k: jp.zeros(v) for k, v in observation_size.items()}, std_eps=1e-6
    )
    return [normalizer, policy, None]
