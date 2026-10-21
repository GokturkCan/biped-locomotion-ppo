"""Resolve runtime names; semantic roles carry no source robot name bindings."""


def role_index(spec, role):
    return spec.joint_names.index(spec.roles[role])


def resolve_ids(model, spec):
    import numpy as np

    joints = np.array([model.joint(n).id for n in spec.joint_names])
    actuators = np.array([model.actuator(n).id for n in spec.actuator_names])
    return joints, actuators
