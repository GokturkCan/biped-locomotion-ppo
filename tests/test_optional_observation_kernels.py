"""Retained observation kernels on fabricated state, without model loading."""

import json
from types import SimpleNamespace
from importlib import resources
import numpy as np
import pytest

for name in ("jax", "flax", "mujoco", "ml_collections"):
    pytest.importorskip(name)
import jax.numpy as jp
from biped_locomotion.environments.base import source_function
from biped_locomotion.commands import process as ucp
from biped_locomotion.observations.u13 import append_unified_context


class FabricatedStateReader:
    def __init__(self):
        defaults = json.loads(
            resources.files("biped_locomotion")
            .joinpath("data/u13_environment.json")
            .read_text()
        )
        self._config = SimpleNamespace(unified=SimpleNamespace(**defaults["unified"]))
        self._qposadr_jp = jp.arange(7, 19)
        self._dofadr_jp = jp.arange(6, 18)
        self._default_pose = jp.zeros(12)
        self._effort_uppers = jp.ones(12)
        self._knee_l_idx = 3
        self._knee_r_idx = 9
        self._enable_sign_specific_yaw_teacher = False
        self._actor_wz_observation_scale = jp.float32(1)
        self._enable_gait_phase_observation = True
        self._u13_neutralize_gait_phase_observation = True
        self._u6v9_corrected_gait_phase = False
        self._routing_enabled = False
        self._dense_delta = jp.array([0.015, 0.015, 0.02])

    def _foot_contacts(self, data):
        return jp.float32(1), jp.float32(0)

    def get_gravity(self, data):
        return jp.array([0.0, 0.0, 1.0])

    def get_local_linvel(self, data):
        return jp.array([0.2, 0.03, 0.0])

    def get_gyro(self, data):
        return jp.array([0.0, 0.0, 0.07])

    def get_orientation(self, data):
        return jp.array([1.0, 0.0, 0.0, 0.0])

    def _forward_yaw_from_quat(self, q):
        return jp.float32(0)

    def get_root_pos(self, data):
        return jp.array([0.0, 0.02, 1.0])

    def get_global_linvel(self, data):
        return jp.array([11.0, 12.0, 13.0])

    def get_global_angvel(self, data):
        return jp.array([21.0, 22.0, 23.0])

    def _foot_vertical_ground_reaction_forces(self, data):
        return jp.float32(7), jp.float32(3)

    def _get_privileged_obs(self, data, obs):
        return source_function("WalkingW1._get_privileged_obs")(self, data, obs)

    def _invoke_source(self, key, *args):
        return source_function(key)(self, *args)


def fabricated():
    data = SimpleNamespace(
        qpos=jp.arange(19, dtype=jp.float32) / 100,
        qvel=jp.zeros(18),
        actuator_force=jp.arange(12, dtype=jp.float32),
    )
    info = dict(
        command=jp.array([0.08, 0.0, 0.0]),
        last_act=jp.zeros(12),
        gait_phase=jp.float32(0.25),
        v24_last_left_placement=jp.float32(0.03),
        v24_last_right_placement=jp.float32(0.04),
        v24_pair_valid=jp.float32(1),
        v24_abs_imbalance_previous=jp.float32(0.1),
    )
    return data, info


def test_retained_v24_observation_dimensions():
    env = FabricatedStateReader()
    data, info = fabricated()
    obs = source_function("ResidualForwardWalkingV24._get_obs")(env, data, info)
    assert obs["state"].shape == (57,) and obs["privileged_state"].shape == (82,)
    np.testing.assert_array_equal(obs["privileged_state"][54:57], [11, 12, 13])


def test_retained_u13_context_and_reserved_block_match_public_helper():
    env = FabricatedStateReader()
    data, info = fabricated()
    command = ucp.neutral_command_state()._replace(
        command=jp.array([0.08, 0.03, 0.10]),
        locomotion_request=jp.float32(1),
        start_swing_side=jp.float32(-1),
        ticks_since_request_change=jp.int32(12),
    )
    info.update(ucp.to_info(command))
    info.update(
        u6_last_credited_side=jp.float32(1), u6_commit_outstanding=jp.float32(1)
    )
    obs = source_function("SchaftUnifiedJoystickV1._get_obs")(env, data, info)
    assert obs["state"].shape == (64,) and obs["privileged_state"].shape == (89,)
    expected_context = np.array(
        [1, np.hypot(0.08, 0.03) / env._config.unified.s_speed, -1, 12 / 25]
    )
    np.testing.assert_allclose(obs["state"][57:61], expected_context, rtol=1e-6)
    np.testing.assert_array_equal(obs["state"][61:64], np.zeros(3))
    np.testing.assert_array_equal(obs["privileged_state"][86:89], np.zeros(3))
    # Convert only the privilege layout for the public helper, never model parameters.
    c = obs["privileged_state"]
    actor = obs["state"]
    v24_layout = np.concatenate([actor[:52], c[61:83], actor[52:57], c[83:86]])
    assembled = append_unified_context(actor[:57], v24_layout, actor[57:61])
    for key in assembled:
        np.testing.assert_allclose(assembled[key], obs[key])
