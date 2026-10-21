import numpy as np
import pytest
from pathlib import Path
from biped_locomotion.training.initialization import fresh_head, ScratchContract
from biped_locomotion.training.ppo_config import load_config


def test_fresh_zero_mean_and_softplus_scale():
    std = np.full(12, 0.27)
    k, b = fresh_head(np.ones((7, 24)), np.ones(24), std)
    np.testing.assert_equal(k, 0)
    np.testing.assert_equal(b[:12], 0)
    np.testing.assert_allclose(np.logaddexp(0, b[12:]) + 0.001, std)
    ScratchContract().validate()
    with pytest.raises(ValueError):
        ScratchContract(fresh_normalizer=False).validate()


def test_resolved_profiles():
    root = Path(__file__).resolve().parents[1]
    v = load_config(root / "configs/v24_forward.yaml")
    u = load_config(root / "configs/u13_scratch.yaml")
    assert (v.actor_size, v.critic_size, v.clipping_epsilon) == (57, 82, 0.30)
    assert (u.actor_size, u.critic_size, u.clipping_epsilon, u.num_timesteps) == (
        64,
        89,
        0.20,
        20480000,
    )


def test_all_resolved_ppo_parameters():
    root = Path(__file__).resolve().parents[1]
    common = dict(
        num_envs=1024,
        num_minibatches=16,
        num_updates_per_batch=4,
        learning_rate=2e-4,
        discounting=0.97,
        gae_lambda=0.95,
        entropy_cost=0.01,
    )
    for filename, overrides in (
        ("v24_forward", dict(unroll_length=10, batch_size=256, clipping_epsilon=0.30)),
        ("u13_scratch", dict(unroll_length=20, batch_size=128, clipping_epsilon=0.20)),
    ):
        config = load_config(root / "configs" / (filename + ".yaml"))
        for name, value in (common | overrides).items():
            assert getattr(config, name) == value
