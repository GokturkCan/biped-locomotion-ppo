"""External-model MJX boundary and explicit source-method dispatch."""

from functools import lru_cache
from importlib import import_module, resources
import json


@lru_cache(None)
def dispatch():
    return json.loads(
        resources.files("biped_locomotion.environments")
        .joinpath("dispatch.json")
        .read_text()
    )


@lru_cache(None)
def source_function(key):
    module, name = dispatch()["registry"][key]
    return getattr(import_module("biped_locomotion." + module), name)


def install_methods(cls, profile):
    for name, key in dispatch()["effective"][profile].items():
        if name == "__init__":
            continue
        fn = source_function(key)
        setattr(cls, name, staticmethod(fn) if key in dispatch()["static"] else fn)
    for name, value in dispatch()["attributes"][profile].items():
        setattr(cls, name, tuple(value) if isinstance(value, list) else value)


class ExternalModelEnvironment:
    def __init__(self, spec, profile, overrides=None):
        import numpy as np
        import jax.numpy as jp
        import mujoco
        from mujoco import mjx
        from ml_collections import config_dict
        from biped_locomotion.model.validation import validate_model

        self._spec = self.spec = spec
        data = json.loads(
            resources.files("biped_locomotion")
            .joinpath(f"data/{profile}_environment.json")
            .read_text()
        )
        cfg = config_dict.ConfigDict(data)
        cfg.update_from_flattened_dict(dict(spec.environment_metadata))
        cfg.action_scale = list(spec.action_scale)
        if overrides:
            cfg.update_from_flattened_dict(overrides)
        for name in ("target_height", "min_root_height"):
            if name not in cfg:
                raise ValueError(f"ModelSpec.environment_metadata must provide {name}")
        if profile == "u13" and (
            cfg.unified.enable_sign_specific_yaw_teacher
            or cfg.unified.yaw_teacher_blend != 0
        ):
            raise ValueError("Only the teacher-free U13 profile is supported")
        self._config = cfg.lock()
        self._ctrl_dt, self._sim_dt = cfg.ctrl_dt, cfg.sim_dt
        ratio = cfg.ctrl_dt / cfg.sim_dt
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("integer physics substeps required")
        self._xml_path = str(spec.model_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        validate_model(self._mj_model, spec)
        m = self._mj_model
        m.opt.timestep = cfg.sim_dt
        m.opt.ccd_iterations = 10
        self._joint_ids = np.array([m.joint(n).id for n in spec.joint_names])
        self._actuator_ids = np.array([m.actuator(n).id for n in spec.actuator_names])
        self._qposadr = m.jnt_qposadr[self._joint_ids].copy()
        self._dofadr = m.jnt_dofadr[self._joint_ids].copy()
        self._reference_qpos = np.asarray(spec.reference_qpos, dtype=float).copy()
        for joint_name, value in spec.reference_pose_mapping.items():
            self._reference_qpos[m.jnt_qposadr[m.joint(joint_name).id]] = value
        self._joint_range = m.jnt_range[self._joint_ids].copy()
        self._effort_range = np.stack(
            [-np.asarray(spec.effort_limits), spec.effort_limits], axis=-1
        )
        self._kp, self._kd = np.asarray(spec.kp), np.asarray(spec.kd)
        self._action_scale = np.asarray(spec.action_scale)
        for i, aid in enumerate(self._actuator_ids):
            m.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_gainprm[aid, :] = 0
            m.actuator_biasprm[aid, :] = 0
            m.actuator_gainprm[aid, 0] = self._kp[i]
            m.actuator_biasprm[aid, 1] = -self._kp[i]
            m.actuator_biasprm[aid, 2] = -self._kd[i]
            m.actuator_ctrllimited[aid] = 0
            m.actuator_forcelimited[aid] = 1
            m.actuator_forcerange[aid, :] = self._effort_range[i]
        keep = {spec.floor_geom, *spec.foot_geoms}
        for gid in range(m.ngeom):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) not in keep:
                m.geom_contype[gid] = 0
                m.geom_conaffinity[gid] = 0
        self._mjx_model = mjx.put_model(m, impl=cfg.impl)
        self._post_init()
        self._total_model_mass = float(np.sum(m.body_mass))
        self._gravity_magnitude = float(abs(m.opt.gravity[2]))
        self._expected_weight_jp = jp.float32(
            self._total_model_mass * self._gravity_magnitude
        )
        self._contact_dist_threshold_jp = jp.float32(cfg.contact_dist_threshold)
        self._load_transfer_force_epsilon_jp = jp.float32(
            cfg.load_transfer_force_epsilon
        )
        self._load_transfer_support_force_fraction_jp = jp.float32(
            cfg.load_transfer_support_force_fraction
        )

    def _invoke_source(self, key, *args, **kwargs):
        fn = source_function(key)
        return (
            fn(*args, **kwargs)
            if key in dispatch()["static"]
            else fn(self, *args, **kwargs)
        )

    def _sensor(self, role, data):
        from .mjx_support import get_sensor_data

        return get_sensor_data(self.mj_model, data, self._spec.sensors[role])

    def get_orientation(self, data):
        return self._sensor("orientation", data)

    def get_gravity(self, data):
        return self._sensor("gravity", data)

    def get_global_linvel(self, data):
        return self._sensor("global_linvel", data)

    def get_global_angvel(self, data):
        return self._sensor("global_angvel", data)

    def get_local_linvel(self, data):
        return self._sensor("local_linvel", data)

    def get_gyro(self, data):
        return self._sensor("gyro", data)

    def get_root_pos(self, data):
        return self._sensor("root_pos", data)

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self):
        return 12

    @property
    def observation_size(self):
        return {"state": (self.actor_size,), "privileged_state": (self.critic_size,)}

    @property
    def dt(self):
        return self._ctrl_dt

    @property
    def sim_dt(self):
        return self._sim_dt

    @property
    def n_substeps(self):
        return round(self.dt / self.sim_dt)

    @property
    def unwrapped(self):
        return self
