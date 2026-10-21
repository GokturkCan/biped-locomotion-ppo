"""Small network initialization/inference only. No optimizer or PPO steps."""

import pytest

for name in ("jax", "flax", "mujoco", "ml_collections", "brax"):
    pytest.importorskip(name)
import jax
import jax.numpy as jp
import numpy as np
from biped_locomotion.training.runner import network_factory
from biped_locomotion.training.initialization import initialize_u13
from biped_locomotion.rewards.unified_contract import U13_INIT_ACTION_STD


def test_asymmetric_network_dimensions_and_fresh_u13():
    net = network_factory("u13")({"state": (64,), "privileged_state": (89,)}, 12)
    normalizer, policy, value = initialize_u13(
        net, {"state": (64,), "privileged_state": (89,)}, U13_INIT_ACTION_STD, 7
    )
    assert (
        value is None
        and int(normalizer.count.hi) == 0
        and int(normalizer.count.lo) == 0
    )
    obs = {"state": jp.ones(64), "privileged_state": jp.ones(89)}
    logits = net.policy_network.apply(normalizer, policy, obs)
    assert logits.shape == (24,)
    np.testing.assert_array_equal(np.asarray(logits[:12]), np.zeros(12))
    np.testing.assert_allclose(
        jax.nn.softplus(logits[12:]) + 0.001, U13_INIT_ACTION_STD, rtol=1e-6
    )
    value_params = net.value_network.init(jax.random.PRNGKey(3))
    assert net.value_network.apply(normalizer, value_params, obs).shape == ()
    v24 = network_factory("v24")({"state": (57,), "privileged_state": (82,)}, 12)
    assert v24.parametric_action_distribution.param_size == 24
