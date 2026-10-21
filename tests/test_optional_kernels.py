"""No physics: compare pure retained kernels on fabricated scalar state."""

from types import SimpleNamespace
from collections import defaultdict
import pytest
import numpy as np

for name in ("jax", "flax", "mujoco", "ml_collections"):
    pytest.importorskip(name)
import jax.numpy as jp
from biped_locomotion.environments.base import source_function
from biped_locomotion.gait.events import qualify_foot


def test_numpy_and_jax_event_kernel_agree():
    cfg = SimpleNamespace(
        v8_liftoff_no_contact_steps=3,
        v8_liftoff_height_gain=0.017,
        v8_touchdown_contact_confirm_steps=2,
        v8_min_steps_between_qualified_touchdowns=4,
    )
    fn = source_function("ResidualForwardWalkingV8._v8_liftoff_touchdown_state")
    for contact in (0, 1):
        for streak in (0, 2, 4):
            info = {
                f"v8_{k}_left": np.float32(v)
                for k, v in {
                    "consecutive_no_contact_steps": streak,
                    "max_height_gain_during_air": 0.02,
                    "air_qualified": 0,
                    "consecutive_contact_steps": 1,
                    "steps_since_last_qualified_touchdown": 9,
                }.items()
            }
            args = (
                "left",
                np.float32(contact),
                np.float32(1),
                np.float32(0.02),
                np.float32(1),
                np.float32(1),
                info,
            )
            a = qualify_foot(cfg, *args)
            b = fn(SimpleNamespace(_config=cfg), *args)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=1e-7)


def test_expected_side_advances_on_transfer_not_raw_touchdown():
    f = source_function("ResidualForwardWalkingV9._v9_components")
    baseline = defaultdict(lambda: jp.float32(0))
    baseline.update(
        {
            k: jp.float32(1)
            for k in (
                "v7_upright_gate",
                "v7_root_height_gate",
                "v7_angular_stability_gate",
                "valid_ground_load_gate",
            )
        }
    )
    baseline.update(
        eligible_touchdown=jp.float32(1),
        qualified_touchdown=jp.float32(1),
        qualified_touchdown_left=jp.float32(1),
        lambda_left=jp.float32(0.6),
        lambda_right=jp.float32(0.4),
    )
    env = SimpleNamespace(
        _config=SimpleNamespace(
            v8_recent_eligible_touchdown_steps=3, v8_support_transfer_steps=5
        ),
        _invoke_source=lambda *args: baseline.copy(),
        _v9_expected_side_gates=lambda info: (jp.float32(1), jp.float32(0)),
        _v9_positive_step_contributions=lambda *args: defaultdict(
            lambda: jp.float32(0)
        ),
        get_gyro=lambda data: jp.zeros(3),
    )
    info = defaultdict(lambda: jp.float32(0))
    info["v9_expected_side"] = jp.float32(-1)
    motion = {"left_contact": jp.float32(1), "right_contact": jp.float32(1)}
    out = f(env, None, info, jp.int32(0), {}, motion)
    assert out["support_transfer_completion"] == 1 and out["next_v9_expected_side"] == 1
    baseline["lambda_left"] = jp.float32(0.45)
    baseline["lambda_right"] = jp.float32(0.55)
    out = f(env, None, info, jp.int32(0), {}, motion)
    assert (
        out["support_transfer_completion"] == 0 and out["next_v9_expected_side"] == -1
    )
