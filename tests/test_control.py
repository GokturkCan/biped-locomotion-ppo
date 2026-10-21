import numpy as np
import pytest
from biped_locomotion.control.residual_position import effective_scale, residual_target
from biped_locomotion.control.pd_actuation import affine_pd


def test_phase_authority_and_clipping():
    s = effective_scale(
        np.ones(12), 0.25, np.r_[np.ones(6), np.zeros(6)], 1.0, np.full(12, 2.0)
    )
    np.testing.assert_equal(s, np.r_[np.full(6, 2.0), np.full(6, 0.25)])
    t = residual_target(
        np.full(12, 3.0), np.zeros(12), s, np.full(12, -0.5), np.full(12, 0.5)
    )
    np.testing.assert_equal(t["target"], np.r_[np.full(6, 0.5), np.full(6, 0.25)])
    assert t["target_clipped"]


def test_pd_units_and_force_clamp():
    np.testing.assert_equal(
        affine_pd([1, -1], [0, 0], [1, -1], [4, 4], [1, 1], [2, 2]), [2, -2]
    )


def test_invalid_vectors_fail():
    with pytest.raises(ValueError):
        residual_target([float("nan")], [0], [1], [-1], [1])


def test_canonical_control_dispatch_uses_fixed_scale_override():
    import json
    from importlib import resources

    table = json.loads(
        resources.files("biped_locomotion.environments")
        .joinpath("dispatch.json")
        .read_text()
    )
    for profile in ("v24", "u13"):
        assert (
            table["effective"][profile]["_q_ref_phase"]
            == "ResidualForwardWalkingV2._q_ref_phase"
        )
        assert (
            table["effective"][profile]["_v3_control_law"]
            == "ResidualForwardWalkingV2._v3_control_law"
        )
