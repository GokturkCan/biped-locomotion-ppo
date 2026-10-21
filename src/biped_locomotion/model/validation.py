"""Fail before physics when external bindings violate the research contracts."""

import numpy as np

SENSOR_WIDTHS = {
    "orientation": 4,
    "gravity": 3,
    "global_linvel": 3,
    "global_angvel": 3,
    "local_linvel": 3,
    "gyro": 3,
    "root_pos": 3,
}


def validate_spec(spec):
    if len(spec.joint_names) != 12 or len(set(spec.joint_names)) != 12:
        raise ValueError("12 unique runtime joint names are required")
    if len(spec.actuator_names) != 12 or len(set(spec.actuator_names)) != 12:
        raise ValueError("12 unique runtime actuator names are required")
    for key in ("kp", "kd", "effort_limits", "action_scale"):
        a = np.asarray(getattr(spec, key))
        if a.shape != (12,) or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"{key}: expected 12 finite nonnegative values")
    if not set(spec.reference_pose_mapping) <= set(spec.joint_names) or not all(
        np.isfinite(v) for v in spec.reference_pose_mapping.values()
    ):
        raise ValueError(
            "reference pose overrides must name supplied joints and be finite"
        )
    if not np.isfinite(spec.reference_qpos).all():
        raise ValueError("reference_qpos must be finite")
    required_roles = {
        side + "_" + joint
        for side in ("left", "right")
        for joint in (
            "hip_yaw",
            "hip_roll",
            "hip_pitch",
            "knee_extension",
            "ankle_pitch",
            "ankle_roll",
        )
    }
    if set(spec.roles) != required_roles or len(set(spec.roles.values())) != 12:
        raise ValueError("12 unique semantic role bindings are required")
    if any(x not in spec.joint_names for x in spec.roles.values()):
        raise ValueError("roles must refer to supplied joint names")
    if not {"left", "right"} <= set(spec.contact_geoms) or not {"left", "right"} <= set(
        spec.contact_bodies
    ):
        raise ValueError("left/right contact geom and body bindings are required")
    if not set(SENSOR_WIDTHS) <= set(spec.sensors):
        raise ValueError("missing sensor bindings")


def validate_model(model, spec):
    validate_spec(spec)
    import mujoco

    if model.nu != 12 or model.nq != len(spec.reference_qpos):
        raise ValueError(
            "model action/reference dimensions do not match runtime metadata"
        )
    for name in spec.joint_names:
        joint = model.joint(name)
        if model.jnt_type[joint.id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError("controlled joints must have one degree of freedom")
        if not model.jnt_limited[joint.id]:
            raise ValueError(
                "explicit joint bounds must be provided by the external model"
            )
    if [model.actuator(n).id for n in spec.actuator_names] != list(range(12)):
        raise ValueError(
            "runtime actuator order must match model actuator storage order"
        )
    for joint_name, actuator_name in zip(spec.joint_names, spec.actuator_names):
        aid = model.actuator(actuator_name).id
        if model.actuator_trnid[aid, 0] != model.joint(joint_name).id:
            raise ValueError(
                "actuator order/transmission does not match supplied joint order"
            )
        if not np.array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
            raise ValueError(
                "affine-PD contract requires direct unit joint transmission"
            )
    for role, width in SENSOR_WIDTHS.items():
        if model.sensor_dim[model.sensor(spec.sensors[role]).id] != width:
            raise ValueError(f"sensor {role} width must be {width}")
    model.body(spec.root_body)
    model.geom(spec.floor_geom)
    for n in spec.foot_geoms:
        model.geom(n)
    for n in spec.contact_bodies.values():
        model.body(n)
