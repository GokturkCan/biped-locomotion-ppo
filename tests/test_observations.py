import numpy as np
from biped_locomotion.observations.common import core_actor, privileged_prefix
from biped_locomotion.observations.v24 import assemble, GROUPS
from biped_locomotion.observations.u13 import append_unified_context


def test_group_order_and_privileged_separation():
    q = np.arange(12, dtype=float)
    z = np.zeros(12)
    core = core_actor(
        q, z, z, [0, 0, 1], [0.2, 0, 0], [0, 0, 0.1], 0.25, z, [1, 0], [0.1, 0, 0]
    )
    priv = privileged_prefix(
        core, [1, 0], [0.3, 0, 0], [0, 0, 0.2], np.ones(12), [2, 3]
    )
    obs = assemble(core, priv, 0.02, 0, 0.03, 0.04, 1, 0.1)
    assert obs["state"].shape == (57,) and obs["privileged_state"].shape == (82,)
    np.testing.assert_equal(obs["state"][:12], q)
    np.testing.assert_equal(obs["privileged_state"][60:72], np.ones(12))
    assert GROUPS["command"] == (49, 52)
    u = append_unified_context(obs["state"], obs["privileged_state"], [0, 1, 0, 0])
    assert u["state"].shape == (64,) and u["privileged_state"].shape == (89,)
    np.testing.assert_equal(u["state"][-3:], 0)


def test_u13_critic_context_precedes_privileged_channels():
    from biped_locomotion.observations.u13 import append_unified_context

    actor = np.arange(57, dtype=float)
    critic_v24_layout = np.arange(82, dtype=float) + 1000
    context = np.array([1.0, 0.4, -1.0, 0.8])
    result = append_unified_context(actor, critic_v24_layout, context)
    np.testing.assert_array_equal(result["state"][57:61], context)
    np.testing.assert_array_equal(result["state"][61:64], np.zeros(3))
    c = result["privileged_state"]
    np.testing.assert_array_equal(c[:57], actor)
    np.testing.assert_array_equal(c[57:61], context)
    np.testing.assert_array_equal(c[61:83], critic_v24_layout[52:74])
    np.testing.assert_array_equal(c[83:86], critic_v24_layout[79:82])
    np.testing.assert_array_equal(c[86:89], np.zeros(3))
