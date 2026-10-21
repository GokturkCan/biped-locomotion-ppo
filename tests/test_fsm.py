"""Tests the extracted source kernel, using fabricated scalar state only."""

from types import SimpleNamespace
import pytest

np = pytest.importorskip("numpy")
import numpy as jp
from biped_locomotion.gait.events import qualify_foot

KEY = "ResidualForwardWalkingV8._v8_liftoff_touchdown_state"


def test_liftoff_and_touchdown_debounce_one_shot():
    f = qualify_foot
    env = SimpleNamespace(
        _config=SimpleNamespace(
            v8_liftoff_no_contact_steps=3,
            v8_liftoff_height_gain=0.017,
            v8_touchdown_contact_confirm_steps=2,
            v8_min_steps_between_qualified_touchdowns=4,
        )
    )
    info = {
        f"v8_{k}_left": jp.float32(v)
        for k, v in {
            "consecutive_no_contact_steps": 0,
            "max_height_gain_during_air": 0,
            "air_qualified": 0,
            "consecutive_contact_steps": 0,
            "steps_since_last_qualified_touchdown": 9,
        }.items()
    }

    def tick(contact, previous):
        result = f(
            env._config,
            "left",
            jp.float32(contact),
            jp.float32(1),
            jp.float32(0.02),
            jp.float32(previous),
            jp.float32(1),
            info,
        )
        for a, b in [
            ("consecutive_no_contact_steps", "next_no_contact_streak"),
            ("max_height_gain_during_air", "next_max_air_height"),
            ("air_qualified", "next_air_qualified"),
            ("consecutive_contact_steps", "next_contact_streak"),
            ("steps_since_last_qualified_touchdown", "next_steps_since"),
        ]:
            info[f"v8_{a}_left"] = result[b]
        return result

    assert tick(0, 1)["qualified_liftoff"] == 0
    assert tick(0, 0)["qualified_liftoff"] == 0
    assert tick(0, 0)["qualified_liftoff"] == 1
    assert tick(0, 0)["qualified_liftoff"] == 0
    assert tick(1, 0)["qualified_touchdown"] == 0
    assert tick(1, 1)["qualified_touchdown"] == 1
    assert tick(1, 1)["qualified_touchdown"] == 0


def test_invalid_ground_load_blocks_qualification():
    f = qualify_foot
    env = SimpleNamespace(
        _config=SimpleNamespace(
            v8_liftoff_no_contact_steps=1,
            v8_liftoff_height_gain=0.017,
            v8_touchdown_contact_confirm_steps=2,
            v8_min_steps_between_qualified_touchdowns=4,
        )
    )
    info = {
        f"v8_{k}_left": jp.float32(v)
        for k, v in {
            "consecutive_no_contact_steps": 5,
            "max_height_gain_during_air": 0.02,
            "air_qualified": 0,
            "consecutive_contact_steps": 0,
            "steps_since_last_qualified_touchdown": 9,
        }.items()
    }
    r = f(
        env._config,
        "left",
        jp.float32(0),
        jp.float32(1),
        jp.float32(0.02),
        jp.float32(1),
        jp.float32(0),
        info,
    )
    assert r["qualified_liftoff"] == 0
