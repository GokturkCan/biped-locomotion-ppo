"""Resolved canonical PPO settings; parsing does not load JAX or a model."""

from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass(frozen=True)
class PPOConfig:
    profile: str = "v24"
    num_envs: int = 1024
    unroll_length: int = 10
    batch_size: int = 256
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    learning_rate: float = 2e-4
    discounting: float = 0.97
    gae_lambda: float = 0.95
    entropy_cost: float = 0.01
    clipping_epsilon: float = 0.30
    episode_length: int = 3000
    num_evals: int = 7
    num_timesteps: int = 3440640
    normalize_observations: bool = True
    max_grad_norm: float = 1.0
    num_resets_per_eval: int = 0
    num_eval_envs: int = 128
    action_repeat: int = 1
    reward_scaling: float = 1.0
    log_training_metrics: bool = True

    def __post_init__(self):
        if self.profile not in ("v24", "u13"):
            raise ValueError("unknown profile")
        for k in (
            "num_envs",
            "unroll_length",
            "batch_size",
            "num_minibatches",
            "num_updates_per_batch",
            "episode_length",
            "num_evals",
            "num_timesteps",
        ):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(f"invalid {k}")
        if (self.batch_size * self.num_minibatches) % self.num_envs:
            raise ValueError("invalid PPO batch geometry")
        if (
            not 0 < self.clipping_epsilon < 1
            or not 0 < self.discounting <= 1
            or not 0 < self.gae_lambda <= 1
            or self.learning_rate <= 0
        ):
            raise ValueError("invalid PPO optimization parameters")

    @property
    def actor_size(self):
        return 57 if self.profile == "v24" else 64

    @property
    def critic_size(self):
        return 82 if self.profile == "v24" else 89

    def train_kwargs(self):
        d = asdict(self)
        d.pop("profile")
        return d


def load_config(path):
    d = yaml.safe_load(Path(path).read_text())
    return PPOConfig(**d["ppo"])
