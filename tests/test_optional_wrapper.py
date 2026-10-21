"""Cached-data auto-reset using a fabricated dataclass state, without MuJoCo."""

import pytest

for name in ("jax", "flax", "mujoco", "ml_collections", "brax"):
    pytest.importorskip(name)
import jax
import jax.numpy as jp
from biped_locomotion.environments.mjx_support import State
from biped_locomotion.training.wrappers import BraxAutoResetWrapper


class FabricatedEnvironment:
    def reset(self, rng):
        return State(
            jp.zeros((2, 1)),
            jp.zeros((2, 1)),
            jp.zeros(2),
            jp.zeros(2),
            {},
            {"steps": jp.zeros(2)},
        )

    def step(self, state, action):
        return state.replace(
            data=state.data + 1, obs=state.obs + 1, done=jp.array([1.0, 0.0])
        )


def test_reset_uses_mjx_data_field():
    env = BraxAutoResetWrapper(FabricatedEnvironment())
    s = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    s = env.step(s, jp.zeros((2, 12)))
    assert float(s.data[0, 0]) == 0 and float(s.data[1, 0]) == 1
    assert s.info["AutoResetWrapper_done_count"].tolist() == [1, 0]
