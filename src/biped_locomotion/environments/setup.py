# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
import jax.numpy as jp


def FootUnloadV2__post_init(self) -> None:
    self._invoke_source("WalkingW1._post_init")
    self._hip_aa_l_idx = self._spec.joint_names.index(self._spec.roles["left_hip_roll"])
    self._hip_aa_r_idx = self._spec.joint_names.index(
        self._spec.roles["right_hip_roll"]
    )
    self._ankle_aa_l_idx = self._spec.joint_names.index(
        self._spec.roles["left_ankle_roll"]
    )
    self._ankle_aa_r_idx = self._spec.joint_names.index(
        self._spec.roles["right_ankle_roll"]
    )


def ResidualForwardWalkingV1__post_init(self) -> None:
    self._invoke_source("ResidualSteppingInPlaceV32._post_init")


def ResidualSteppingInPlaceV1__post_init(self) -> None:
    self._invoke_source("FootUnloadV2._post_init")
    self._half_cycle_steps = int(
        self._config.bridge_prep_shift_steps
        + self._config.bridge_unload_steps
        + self._config.bridge_swing_hold_steps
        + self._config.bridge_touchdown_steps
        + self._config.bridge_recenter_steps
    )
    self._cycle_steps = 2 * self._half_cycle_steps
    left_leg = [
        1.0
        if name in {v for k, v in self._spec.roles.items() if k.startswith("left_")}
        else 0.0
        for name in self._spec.joint_names
    ]
    right_leg = [
        1.0
        if name in {v for k, v in self._spec.roles.items() if k.startswith("right_")}
        else 0.0
        for name in self._spec.joint_names
    ]
    self._left_leg_mask_jp = jp.array(left_leg)
    self._right_leg_mask_jp = jp.array(right_leg)


def ResidualSteppingInPlaceV3__post_init(self) -> None:
    self._invoke_source("ResidualSteppingInPlaceV1._post_init")
    self._floor_geom_id = self._mj_model.geom(self._spec.floor_geom).id


def ResidualSteppingInPlaceV32__post_init(self) -> None:
    self._invoke_source("ResidualSteppingInPlaceV3._post_init")


def Walking__post_init(self) -> None:
    self._init_q = jp.asarray(self._reference_qpos)
    self._default_pose = self._init_q[jp.array(self._qposadr)]
    self._qposadr_jp = jp.array(self._qposadr)
    self._dofadr_jp = jp.array(self._dofadr)
    self._action_scale_jp = jp.array(self._config.action_scale)
    self._joint_lowers = jp.array(self._joint_range[:, 0])
    self._joint_uppers = jp.array(self._joint_range[:, 1])
    joint_center = 0.5 * (self._joint_lowers + self._joint_uppers)
    joint_radius = self._joint_uppers - self._joint_lowers
    soft_factor = self._config.soft_joint_pos_limit_factor
    self._soft_joint_lowers = joint_center - 0.5 * joint_radius * soft_factor
    self._soft_joint_uppers = joint_center + 0.5 * joint_radius * soft_factor
    self._effort_uppers = jp.array(self._effort_range[:, 1])
    self._knee_l_idx = self._spec.joint_names.index(
        self._spec.roles["left_knee_extension"]
    )
    self._knee_r_idx = self._spec.joint_names.index(
        self._spec.roles["right_knee_extension"]
    )


def WalkingW05__post_init(self) -> None:
    self._invoke_source("Walking._post_init")
    self._floor_geom_id = self._mj_model.geom(self._spec.floor_geom).id
    self._left_foot_geom_id = self._mj_model.geom(self._spec.foot_geoms[0]).id
    self._right_foot_geom_id = self._mj_model.geom(self._spec.foot_geoms[1]).id


def WalkingW1__post_init(self) -> None:
    self._invoke_source("WalkingW05._post_init")
    self._left_foot_body_id = self._mj_model.body(self._spec.contact_bodies["left"]).id
    self._right_foot_body_id = self._mj_model.body(
        self._spec.contact_bodies["right"]
    ).id
