from pathlib import Path
from dataclasses import replace
import pytest
from biped_locomotion.model.interface import ModelSpec
from biped_locomotion.model.validation import validate_spec


def synthetic_spec():
    names = tuple(f"independent_joint_{i}" for i in range(12))
    roles = [
        s + "_" + j
        for s in ("left", "right")
        for j in (
            "hip_yaw",
            "hip_roll",
            "hip_pitch",
            "knee_extension",
            "ankle_pitch",
            "ankle_roll",
        )
    ]
    return ModelSpec(
        Path("not_a_distributed_model"),
        names,
        tuple(f"independent_actuator_{i}" for i in range(12)),
        dict(zip(roles, names)),
        {"left": "fabricated_contact_a", "right": "fabricated_contact_b"},
        {"left": "fabricated_body_a", "right": "fabricated_body_b"},
        "fabricated_floor",
        "fabricated_root",
        dict.fromkeys(
            [
                "orientation",
                "gravity",
                "global_linvel",
                "global_angvel",
                "local_linvel",
                "gyro",
                "root_pos",
            ],
            "runtime_sensor",
        ),
        (0.0,) * 19,
        (1.0,) * 12,
        (0.2,) * 12,
        (8.0,) * 12,
        (0.07,) * 12,
        {"target_height": 1.0, "min_root_height": 0.4},
    )


def test_runtime_metadata_has_no_fallback_bindings():
    s = synthetic_spec()
    validate_spec(s)
    with pytest.raises(ValueError):
        validate_spec(replace(s, roles={}))
    with pytest.raises(ValueError):
        validate_spec(replace(s, action_scale=(1.0,)))
