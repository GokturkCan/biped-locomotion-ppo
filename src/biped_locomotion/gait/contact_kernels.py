# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from mujoco import mjx


def ResidualForwardWalkingV4Force__foot_vertical_ground_reaction_forces(
    self, data: mjx.Data
) -> tuple[jax.Array, jax.Array]:
    contact = data._impl.contact
    efc_force = data._impl.efc_force
    address = jp.asarray(contact.efc_address)
    safe_address = jp.maximum(address, 0)
    active = (address >= 0) & (contact.dist <= self._contact_dist_threshold_jp)
    normal = (
        efc_force[safe_address]
        + efc_force[safe_address + 1]
        + efc_force[safe_address + 2]
        + efc_force[safe_address + 3]
    )
    tangent_1 = (
        efc_force[safe_address] - efc_force[safe_address + 1]
    ) * contact.friction[:, 0]
    tangent_2 = (
        efc_force[safe_address + 2] - efc_force[safe_address + 3]
    ) * contact.friction[:, 1]
    force_contact = jp.stack([normal, tangent_1, tangent_2], axis=-1)
    frame = contact.frame.reshape((-1, 3, 3))
    force_world = jp.einsum("ni,nij->nj", force_contact, frame)
    fz_each = jp.maximum(force_world[:, 2], 0.0)
    geom1 = contact.geom1
    geom2 = contact.geom2
    left_pair = (geom1 == self._floor_geom_id) & (geom2 == self._left_foot_geom_id) | (
        geom2 == self._floor_geom_id
    ) & (geom1 == self._left_foot_geom_id)
    right_pair = (geom1 == self._floor_geom_id) & (
        geom2 == self._right_foot_geom_id
    ) | (geom2 == self._floor_geom_id) & (geom1 == self._right_foot_geom_id)
    fz_left = jp.sum(jp.where(active & left_pair, fz_each, 0.0))
    fz_right = jp.sum(jp.where(active & right_pair, fz_each, 0.0))
    return (fz_left, fz_right)


def ResidualSteppingInPlaceV1__foot_roll_pitch(
    self, data: mjx.Data
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Cheap per-foot roll/pitch from geom_xmat.

    Matrix indexing only (no contact-force decode), the same technique as
    scripts/schaft_bridge_diagnostic.py's _foot_geom_diagnostics -- safe to
    compute every training step, unlike the contact-force-sum diagnostics
    that script deliberately keeps opt-in/standalone.
    """

    def roll_pitch(mat_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        mat = mat_flat.reshape(3, 3)
        r20, r21, r22 = (mat[2, 0], mat[2, 1], mat[2, 2])
        roll = jp.arctan2(r21, r22)
        pitch = jp.arctan2(-r20, jp.sqrt(r21 * r21 + r22 * r22))
        return (roll, pitch)

    left_roll, left_pitch = roll_pitch(data.geom_xmat[self._left_foot_geom_id])
    right_roll, right_pitch = roll_pitch(data.geom_xmat[self._right_foot_geom_id])
    return (left_roll, left_pitch, right_roll, right_pitch)


def ResidualSteppingInPlaceV3__contact_count(
    self, data: mjx.Data, geom_id: int
) -> jax.Array:
    impl = data._impl
    if hasattr(impl, "contact"):
        contact = impl.contact
        geom1, geom2, dist = (contact.geom1, contact.geom2, contact.dist)
    elif hasattr(impl, "contact__geom") and hasattr(impl, "contact__dist"):
        geom_pairs = impl.contact__geom
        geom1, geom2, dist = (geom_pairs[:, 0], geom_pairs[:, 1], impl.contact__dist)
    else:
        return jp.zeros((), dtype=jp.int32)
    active = dist <= 0.001
    pair = (geom1 == self._floor_geom_id) & (geom2 == geom_id) | (
        geom2 == self._floor_geom_id
    ) & (geom1 == geom_id)
    return jp.sum(pair & active)


def WalkingW05__foot_contacts(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    if hasattr(data._impl, "contact"):
        contact = data._impl.contact
        geom1 = contact.geom1
        geom2 = contact.geom2
        dist = contact.dist
    elif hasattr(data._impl, "contact__geom") and hasattr(data._impl, "contact__dist"):
        geom_pairs = data._impl.contact__geom
        geom1 = geom_pairs[:, 0]
        geom2 = geom_pairs[:, 1]
        dist = data._impl.contact__dist
    else:
        return (jp.zeros((), dtype=bool), jp.zeros((), dtype=bool))
    active = dist <= 0.001
    left_pair = (geom1 == self._floor_geom_id) & (geom2 == self._left_foot_geom_id) | (
        geom2 == self._floor_geom_id
    ) & (geom1 == self._left_foot_geom_id)
    right_pair = (geom1 == self._floor_geom_id) & (
        geom2 == self._right_foot_geom_id
    ) | (geom2 == self._floor_geom_id) & (geom1 == self._right_foot_geom_id)
    left_contact = jp.any(left_pair & active)
    right_contact = jp.any(right_pair & active)
    return (left_contact, right_contact)


def WalkingW1__foot_motion(
    self, data: mjx.Data, info: dict[str, Any]
) -> dict[str, jax.Array]:
    left_pos, right_pos = self._foot_positions(data)
    left_contact, right_contact = self._foot_contacts(data)
    left_vel_xy = (left_pos[:2] - info["prev_left_foot_xy"]) / self.dt
    right_vel_xy = (right_pos[:2] - info["prev_right_foot_xy"]) / self.dt
    left_air_time = info["feet_air_time"][0]
    right_air_time = info["feet_air_time"][1]
    air_time_reward = jp.where(
        left_contact & (left_air_time > self._config.min_air_time),
        left_air_time - self._config.min_air_time,
        0.0,
    ) + jp.where(
        right_contact & (right_air_time > self._config.min_air_time),
        right_air_time - self._config.min_air_time,
        0.0,
    )
    feet_slip = jp.float32(left_contact) * jp.sum(jp.square(left_vel_xy)) + jp.float32(
        right_contact
    ) * jp.sum(jp.square(right_vel_xy))
    contacts = jp.array([left_contact, right_contact])
    new_air_time = jp.where(contacts, jp.zeros(2), info["feet_air_time"] + self.dt)
    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "left_pos": left_pos,
        "right_pos": right_pos,
        "left_vel_xy": left_vel_xy,
        "right_vel_xy": right_vel_xy,
        "left_vel_xy_norm": jp.linalg.norm(left_vel_xy),
        "right_vel_xy_norm": jp.linalg.norm(right_vel_xy),
        "feet_air_time": new_air_time,
        "feet_air_time_reward": air_time_reward,
        "feet_slip": feet_slip,
    }


def WalkingW1__foot_positions(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    if hasattr(data, "geom_xpos"):
        return (
            data.geom_xpos[self._left_foot_geom_id],
            data.geom_xpos[self._right_foot_geom_id],
        )
    return (data.xpos[self._left_foot_body_id], data.xpos[self._right_foot_body_id])
