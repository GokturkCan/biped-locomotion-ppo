# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

"""Body/world-frame command transform for the retained unified command/observation contract.

OBS_CONTRACT_V2
---------------
The retained command representation at actor indices 49-51 expresses the desired
command in the robot BODY frame:

    obs[49] = vx_body   (desired forward velocity, body frame)
    obs[50] = vy_body   (desired lateral velocity, body frame)
    obs[51] = yaw_rate  (desired yaw rate; frame-invariant)

External command sources reach that body-frame representation two ways:

  * BODY-frame sources (physical joystick / teleoperation) are already in the
    body frame and are passed through unchanged. They MUST bypass this
    converter; selection between pass-through and conversion belongs to the
    caller, not this transform.
  * WORLD-frame sources (path or navigation commands) are rotated into the body
    frame by this module using ``R(-yaw)``.

Transform (rotation of a world-frame planar velocity into the body frame):

    cx_body = vx_world * cos(yaw) + vy_world * sin(yaw)
    cy_body = -vx_world * sin(yaw) + vy_world * cos(yaw)

This is ``[cx_body, cy_body]^T = R(-yaw) [vx_world, vy_world]^T`` where
``R(theta) = [[cos, -sin], [sin, cos]]``.

Legacy equivalence: at ``vy_world = 0`` this reduces to
``(vx_world*cos(yaw), -vx_world*sin(yaw))``, which is exactly the V22/V24
scalar-forward projection ``(vx*cos(-yaw), vx*sin(-yaw))``. WORLD mode is
therefore analytically equivalent to V24 observation semantics at
``vy_world = 0``; that equivalence is numerically compared using an explicit
float32 tolerance, not asserted bit-for-bit.

This module is intentionally pure: no environment or config imports, no I/O,
no side effects. It supports scalar and batched array inputs and is compatible
with ``jax.jit``.
"""

import jax.numpy as jp


def world_to_body_command(vx_world, vy_world, yaw):
    """Rotate a world-frame planar velocity command into the body frame.

    Computes ``R(-yaw) [vx_world, vy_world]``:

        cx_body = vx_world * cos(yaw) + vy_world * sin(yaw)
        cy_body = -vx_world * sin(yaw) + vy_world * cos(yaw)

    Inputs may be scalars or broadcastable arrays; the output dtype follows JAX's
    standard promotion of the inputs (float32 under the default configuration).
    This function has no side effects and is ``jax.jit``-compatible.

    Args:
      vx_world: Desired world-frame forward (x) velocity.
      vy_world: Desired world-frame lateral (y) velocity.
      yaw: Robot heading (radians), such that ``R(yaw)`` maps body to world.

    Returns:
      A ``(cx_body, cy_body)`` tuple of the command in the body frame.
    """
    cos_yaw = jp.cos(yaw)
    sin_yaw = jp.sin(yaw)
    cx_body = vx_world * cos_yaw + vy_world * sin_yaw
    cy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    return (cx_body, cy_body)


def body_command_vector(vx_world, vy_world, yaw_rate, yaw):
    """World-frame planar command + yaw rate expressed for actor indices 49-51.

    Rotates ``(vx_world, vy_world)`` into the body frame via
    :func:`world_to_body_command` and passes ``yaw_rate`` through unchanged
    (yaw rate is frame-invariant for a planar base). The result maps directly to
    OBS_CONTRACT_V2 actor indices ``[49, 50, 51]``.

    BODY-frame command sources must NOT call this function; they are already in
    the body frame and must be passed through unchanged by the caller.

    Args:
      vx_world: Desired world-frame forward (x) velocity.
      vy_world: Desired world-frame lateral (y) velocity.
      yaw_rate: Desired yaw rate (radians/s); returned unchanged.
      yaw: Robot heading (radians).

    Returns:
      A ``(cx_body, cy_body, yaw_rate)`` tuple for observation indices 49-51.
    """
    cx_body, cy_body = world_to_body_command(vx_world, vy_world, yaw)
    return (cx_body, cy_body, yaw_rate)
