# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

"""Pure, JIT-safe command process and path reference for the unified V1 env.

Every equation in document 10 §8 (command semantics), §9.4 (path reference) and
§12.2 (command-region weights) lives here exactly once. Training, the
deterministic evaluator and the renderer all call these same functions, so a
Python/Python divergence is structurally impossible.

The module holds **no** mutable global state and never touches ``walking.py``.
All state is carried in :class:`CommandState`, a JAX pytree.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jp
from biped_locomotion.rewards import unified_contract as ucfg


class CommandState(NamedTuple):
    """Complete command/gate/path state. A JAX pytree; safe inside ``jit``."""

    command: jax.Array
    command_target: jax.Array
    category: jax.Array
    hold_ticks_remaining: jax.Array
    instant: jax.Array
    pending_command: jax.Array
    pending_category: jax.Array
    pending_hold: jax.Array
    pending_valid: jax.Array
    raw_request: jax.Array
    raw_streak: jax.Array
    locomotion_request: jax.Array
    ticks_since_request_change: jax.Array
    heading_target: jax.Array
    p_ref: jax.Array
    v_world: jax.Array
    start_swing_side: jax.Array
    next_start_swing_side: jax.Array


class SchedulerSpec(NamedTuple):
    """Numeric scheduler parameters, as arrays so they can be traced."""

    category_probs: jax.Array
    forward_vx: jax.Array
    backward_vx: jax.Array
    lateral_vy: jax.Array
    lateral_vx: jax.Array
    lateral_positive_probability: jax.Array
    yaw_vx: jax.Array
    yaw_wz: jax.Array
    yaw_in_place_probability: jax.Array
    yaw_in_place_wz: jax.Array
    mixed_vx: jax.Array
    mixed_vy: jax.Array
    mixed_wz: jax.Array
    mixed_positive_vx_probability: jax.Array
    enable_mixed_composition: jax.Array
    mixed_composition_probs: jax.Array
    mixed_vx_forward: jax.Array
    mixed_vx_backward: jax.Array
    quiet_hold: jax.Array
    moving_hold: jax.Array
    yaw_hold: jax.Array
    category_switch_probability: jax.Array
    instant_transition_prob: jax.Array
    slew: jax.Array
    max_speed: jax.Array
    compat_v_world: jax.Array
    reset_quiet_probability: jax.Array


def spec_from_config(cfg) -> SchedulerSpec:
    """Build a :class:`SchedulerSpec` from a validated unified config."""
    u = cfg.unified
    ctrl_dt = float(cfg.ctrl_dt)
    slew = jp.asarray([float(s) * ctrl_dt for s in u.slew], dtype=jp.float32)
    return SchedulerSpec(
        category_probs=jp.asarray(list(u.category_probs), dtype=jp.float32),
        forward_vx=jp.asarray(list(u.forward_vx), dtype=jp.float32),
        backward_vx=jp.asarray(list(u.backward_vx), dtype=jp.float32),
        lateral_vy=jp.asarray(list(u.lateral_vy), dtype=jp.float32),
        lateral_vx=jp.asarray(list(u.lateral_vx), dtype=jp.float32),
        lateral_positive_probability=jp.asarray(
            float(u.lateral_positive_probability), dtype=jp.float32
        ),
        yaw_vx=jp.asarray(list(u.yaw_vx), dtype=jp.float32),
        yaw_wz=jp.asarray(list(u.yaw_wz), dtype=jp.float32),
        yaw_in_place_probability=jp.asarray(
            float(u.yaw_in_place_probability), dtype=jp.float32
        ),
        yaw_in_place_wz=jp.asarray(list(u.yaw_in_place_wz), dtype=jp.float32),
        mixed_vx=jp.asarray(list(u.mixed_vx), dtype=jp.float32),
        mixed_vy=jp.asarray(list(u.mixed_vy), dtype=jp.float32),
        mixed_wz=jp.asarray(list(u.mixed_wz), dtype=jp.float32),
        mixed_positive_vx_probability=jp.asarray(
            float(u.mixed_positive_vx_probability), dtype=jp.float32
        ),
        enable_mixed_composition=jp.asarray(
            1.0 if bool(u.enable_mixed_composition) else 0.0, dtype=jp.float32
        ),
        mixed_composition_probs=jp.asarray(
            list(u.mixed_composition_probs), dtype=jp.float32
        ),
        mixed_vx_forward=jp.asarray(list(u.mixed_vx_forward), dtype=jp.float32),
        mixed_vx_backward=jp.asarray(list(u.mixed_vx_backward), dtype=jp.float32),
        quiet_hold=jp.asarray(list(u.quiet_hold_ticks), dtype=jp.int32),
        moving_hold=jp.asarray(list(u.moving_hold_ticks), dtype=jp.int32),
        yaw_hold=jp.asarray(list(u.yaw_hold_ticks), dtype=jp.int32),
        category_switch_probability=jp.asarray(
            float(u.category_switch_probability), dtype=jp.float32
        ),
        instant_transition_prob=jp.asarray(
            float(u.instant_transition_prob), dtype=jp.float32
        ),
        slew=slew,
        max_speed=jp.asarray(float(u.envelope_max_speed), dtype=jp.float32),
        compat_v_world=jp.asarray(list(u.compat_v_world_range), dtype=jp.float32),
        reset_quiet_probability=jp.asarray(
            float(u.reset_quiet_probability), dtype=jp.float32
        ),
    )


def wrap_to_pi(a: jax.Array) -> jax.Array:
    """``atan2(sin a, cos a)`` — the §8.2 wrapping convention."""
    return jp.arctan2(jp.sin(a), jp.cos(a))


def rot2d(angle: jax.Array) -> jax.Array:
    """``R(a) = [[cos a, -sin a], [sin a, cos a]]`` (§9.4)."""
    c, s = (jp.cos(angle), jp.sin(angle))
    return jp.array([[c, -s], [s, c]])


def _uniform(rng: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    return jax.random.uniform(rng, (), minval=lo, maxval=hi)


def _random_sign(rng: jax.Array) -> jax.Array:
    return jp.where(jax.random.bernoulli(rng, 0.5), 1.0, -1.0)


def _sample_hold(rng: jax.Array, bounds: jax.Array) -> jax.Array:
    return jax.random.randint(rng, (), bounds[0], bounds[1] + 1)


def _limit_speed(
    vx: jax.Array, vy: jax.Array, max_speed: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Enforce the §8.3 planar speed constraint without changing direction."""
    speed = jp.sqrt(vx * vx + vy * vy)
    factor = jp.where(speed > max_speed, max_speed / jp.maximum(speed, 1e-09), 1.0)
    return (vx * factor, vy * factor)


def sample_category(rng: jax.Array, spec: SchedulerSpec) -> jax.Array:
    """Sample a category. Zero-probability categories are *unreachable*.

    ``log(0) = -inf`` makes the category structurally impossible to emit, which
    is stronger than merely improbable (§8.4).
    """
    logits = jp.log(jp.maximum(spec.category_probs, 0.0))
    return jax.random.categorical(rng, logits).astype(jp.int32)


def _lateral_forward_component(
    lat_mag: jax.Array, lateral_vy: jax.Array, lateral_vx: jax.Array
) -> jax.Array:
    """LATERAL's forward-speed component (U5_UNIFIED_FORWARD_LATERAL_V1).

    Maps ``lat_mag``'s normalized position within ``[lateral_vy_lo,
    lateral_vy_hi]`` affinely onto ``[lateral_vx_lo, lateral_vx_hi]``, so the
    forward component is a deterministic function of the SAME draw that
    already produced the signed lateral magnitude -- no additional
    ``jax.random`` split is consumed, and every existing stage's RNG-driven
    command sequence is therefore untouched by this extension.

    Edge cases (all required so the default ``(0, 0)`` reproduces the
    original pure-lateral ``vx=0`` semantics bit-for-bit, and so a
    degenerate ``lateral_vy`` interval can never divide by zero):
      * ``lateral_vx_lo == lateral_vx_hi`` returns that exact fixed value,
        independent of ``lat_mag`` -- covers the default ``(0, 0)`` case
        (always exactly ``0.0``) and any other fixed-``vx`` stage choice.
      * ``lateral_vy_hi <= lateral_vy_lo`` (a degenerate interval) falls
        back to a fixed denominator of ``1.0`` rather than dividing by zero;
        the normalized position is then clipped into ``[0, 1]`` regardless,
        so the result stays finite and deterministic.
    """
    vx_lo, vx_hi = (lateral_vx[0], lateral_vx[1])
    vy_lo, vy_hi = (lateral_vy[0], lateral_vy[1])
    fixed_vx = vx_lo == vx_hi
    denom = jp.where(vy_hi > vy_lo, vy_hi - vy_lo, 1.0)
    u = jp.clip((lat_mag - vy_lo) / denom, 0.0, 1.0)
    mapped = vx_lo + u * (vx_hi - vx_lo)
    return jp.where(fixed_vx, vx_lo, mapped)


def sample_command_for_category(
    rng: jax.Array, category: jax.Array, spec: SchedulerSpec
) -> jax.Array:
    """Sample a body-frame ``[vx, vy, wz]`` for ``category`` (§8.4, §13)."""
    (
        k_fwd,
        k_bwd,
        k_lat,
        k_yaw_v,
        k_yaw_w,
        k_yaw_s,
        k_mx,
        k_my,
        k_mw,
        k_sx,
        k_sy,
        k_sw,
    ) = jax.random.split(rng, 12)
    fwd = jp.array([_uniform(k_fwd, spec.forward_vx[0], spec.forward_vx[1]), 0.0, 0.0])
    bwd_mag = _uniform(k_bwd, spec.backward_vx[0], spec.backward_vx[1])
    bwd = jp.array([-bwd_mag, 0.0, 0.0])
    lat_mag = _uniform(k_lat, spec.lateral_vy[0], spec.lateral_vy[1])
    lat_vx = _lateral_forward_component(lat_mag, spec.lateral_vy, spec.lateral_vx)
    lat_sign = jp.where(
        jax.random.bernoulli(k_sy, spec.lateral_positive_probability), 1.0, -1.0
    )
    lat = jp.array([lat_vx, lat_mag * lat_sign, 0.0])
    yaw_sign = _random_sign(k_yaw_s)
    yaw_moving = jp.array(
        [
            _uniform(k_yaw_v, spec.yaw_vx[0], spec.yaw_vx[1]),
            0.0,
            _uniform(k_yaw_w, spec.yaw_wz[0], spec.yaw_wz[1]) * yaw_sign,
        ]
    )
    yaw_in_place = jp.array(
        [
            0.0,
            0.0,
            _uniform(k_yaw_w, spec.yaw_in_place_wz[0], spec.yaw_in_place_wz[1])
            * yaw_sign,
        ]
    )
    is_in_place_yaw = jax.random.bernoulli(
        jax.random.fold_in(k_yaw_v, 2038478), spec.yaw_in_place_probability
    )
    yaw = jp.where(is_in_place_yaw, yaw_in_place, yaw_moving)
    mixed_vx_sign = jp.where(
        jax.random.bernoulli(k_sx, spec.mixed_positive_vx_probability), 1.0, -1.0
    )
    mixed_legacy = jp.array(
        [
            _uniform(k_mx, spec.mixed_vx[0], spec.mixed_vx[1]) * mixed_vx_sign,
            _uniform(k_my, spec.mixed_vy[0], spec.mixed_vy[1]) * _random_sign(k_sy),
            _uniform(k_mw, spec.mixed_wz[0], spec.mixed_wz[1]) * _random_sign(k_sw),
        ]
    )
    comp_logits = jp.log(jp.maximum(spec.mixed_composition_probs, 0.0))
    family = jax.random.categorical(jax.random.fold_in(k_mw, 12648430), comp_logits)
    fwd_mag = _uniform(k_mx, spec.mixed_vx_forward[0], spec.mixed_vx_forward[1])
    bwd_mag = _uniform(k_mx, spec.mixed_vx_backward[0], spec.mixed_vx_backward[1])
    comp_vx = jp.where(mixed_vx_sign > 0.0, fwd_mag, -bwd_mag)
    comp_vy = _uniform(k_my, spec.mixed_vy[0], spec.mixed_vy[1]) * _random_sign(k_sy)
    comp_wz = _uniform(k_mw, spec.mixed_wz[0], spec.mixed_wz[1]) * _random_sign(k_sw)
    vx_on = jp.any(jp.array([family == 0, family == 1, family == 3])).astype(jp.float32)
    vy_on = jp.any(jp.array([family == 0, family == 2, family == 3])).astype(jp.float32)
    wz_on = jp.any(jp.array([family == 1, family == 2, family == 3])).astype(jp.float32)
    mixed_composed = jp.array([comp_vx * vx_on, comp_vy * vy_on, comp_wz * wz_on])
    mixed = jp.where(spec.enable_mixed_composition > 0.5, mixed_composed, mixed_legacy)
    quiet = jp.zeros(3)
    options = jp.stack([quiet, fwd, bwd, lat, yaw, mixed])
    cmd = options[category]
    vx, vy = _limit_speed(cmd[0], cmd[1], spec.max_speed)
    is_quiet = category == ucfg.CAT_QUIET
    return jp.where(is_quiet, jp.zeros(3), jp.array([vx, vy, cmd[2]]))


def _sign_reverses(
    current: jax.Array,
    proposed: jax.Array,
    deadband: float = ucfg.QUIET_SPEED_THRESHOLD,
) -> jax.Array:
    """True when vx or vy flips sign between ``current`` and ``proposed``.

    Such a transition must be routed through an explicit QUIET segment rather
    than slewed through the deadband, which would fabricate a stop/start (§8.4).
    """

    def _sgn(v):
        return jp.where(jp.abs(v) > deadband, jp.sign(v), 0.0)

    rev_x = _sgn(current[0]) * _sgn(proposed[0]) < 0.0
    rev_y = _sgn(current[1]) * _sgn(proposed[1]) < 0.0
    return jp.logical_or(rev_x, rev_y)


def quiet_command(command: jax.Array) -> jax.Array:
    """``True`` when the command sits inside the quiet deadband (§8.5)."""
    speed = jp.linalg.norm(command[:2])
    return jp.logical_and(
        speed <= ucfg.QUIET_SPEED_THRESHOLD,
        jp.abs(command[2]) <= ucfg.QUIET_YAW_THRESHOLD,
    )


def update_gate(state: CommandState, command: jax.Array) -> CommandState:
    """Advance the debounced locomotion gate one tick (§8.5)."""
    raw_request = jp.where(quiet_command(command), 0, 1).astype(jp.int32)
    same = raw_request == state.raw_request
    raw_streak = jp.where(same, state.raw_streak + 1, 1).astype(jp.int32)
    accept = raw_streak >= ucfg.DEBOUNCE_TICKS
    prev_gate = state.locomotion_request
    new_gate = jp.where(accept, raw_request.astype(jp.float32), prev_gate)
    flipped = new_gate != prev_gate
    ticks = jp.where(
        flipped,
        0,
        jp.minimum(state.ticks_since_request_change + 1, ucfg.TRANSITION_TICKS),
    ).astype(jp.int32)
    return state._replace(
        raw_request=raw_request,
        raw_streak=raw_streak,
        locomotion_request=new_gate,
        ticks_since_request_change=ticks,
    )


def transition_progress(state: CommandState) -> jax.Array:
    """``ticks_since_request_change / TRANSITION_TICKS`` in ``[0, 1]`` (§8.5)."""
    return state.ticks_since_request_change.astype(jp.float32) / float(
        ucfg.TRANSITION_TICKS
    )


def reference_velocity_world(state: CommandState, compat: bool) -> jax.Array:
    """``v_ref_world`` (§9.4), by stage mode.

    Args:
      state: current command state.
      compat: ``True`` for the U0/U1 compatibility mode, which uses the
        world-forward reference ``[v_world, 0]``; ``False`` for the shipped
        ``R(psi_ref) @ [vx_body, vy_body]``.
    """
    if compat:
        return jp.array([state.v_world, 0.0])
    return rot2d(state.heading_target) @ state.command[:2]


def path_errors(
    p_ref: jax.Array, root_xy: jax.Array, psi_ref: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return ``(e_path_x, e_path_y)`` in the reference frame (§9.4)."""
    e_world = root_xy - p_ref
    e_ref = rot2d(-psi_ref) @ e_world
    return (e_ref[0], e_ref[1])


def forward_progress(
    prev_root_xy: jax.Array, root_xy: jax.Array, psi_ref: jax.Array
) -> jax.Array:
    """Consecutive-position progress along the heading (§9.4).

    Deliberately *not* an integral in a rotating basis: this cannot accrue
    artificial progress from ``psi_ref`` changing alone.
    """
    delta = root_xy - prev_root_xy
    heading_forward = jp.array([jp.cos(psi_ref), jp.sin(psi_ref)])
    return jp.dot(delta, heading_forward)


class RegionWeights(NamedTuple):
    """Command-reference-frame reward-region weights (§12.2)."""

    speed_cmd: jax.Array
    w_move: jax.Array
    w_fwd: jax.Array
    w_bwd: jax.Array
    w_lat: jax.Array
    w_turn: jax.Array
    w_str: jax.Array


def command_region_weights(
    v_ref_world: jax.Array, psi_ref: jax.Array, wz: jax.Array
) -> RegionWeights:
    """Compute ``w_*`` from the command reference frame (§12.2).

    Using ``c_region = R(-psi_ref) @ v_ref_world`` rather than the raw body
    command is what makes U0/U1 reduce *exactly* to ``w_fwd=1, w_lat=0,
    w_bwd=0``; taking the body command directly would yield the yaw-dependent
    ``w_fwd=cos(yaw)``.
    """
    c_region = rot2d(-psi_ref) @ v_ref_world
    speed_cmd = jp.linalg.norm(c_region)
    u_hat = c_region / jp.maximum(speed_cmd, 1e-06)
    w_move = (speed_cmd > ucfg.QUIET_SPEED_THRESHOLD).astype(jp.float32)
    w_fwd = w_move * jp.clip(u_hat[0], 0.0, 1.0)
    w_bwd = w_move * jp.clip(-u_hat[0], 0.0, 1.0)
    w_lat = w_move * jp.clip(jp.abs(u_hat[1]), 0.0, 1.0)
    w_turn = jp.clip(jp.abs(wz) / ucfg.S_WZ, 0.0, 1.0)
    w_str = 1.0 - w_turn
    return RegionWeights(speed_cmd, w_move, w_fwd, w_bwd, w_lat, w_turn, w_str)


class DenseAxisTerms(NamedTuple):
    """One axis's ``(track, prog, wrong)`` dense-tracking triple (§12.3)."""

    track: jax.Array
    prog: jax.Array
    wrong: jax.Array


def dense_signed_tracking(
    measured: jax.Array, commanded: jax.Array, delta: jax.Array, sigma: jax.Array
) -> DenseAxisTerms:
    """The frozen §12.3 per-axis dense kernel, exactly as document 10 states it.

    Symmetric by construction under ``(commanded, measured) ->
    (-commanded, -measured)``: ``e = measured - commanded`` negates (so
    ``(e/sigma)**2`` is invariant) and ``signed = measured * sign(commanded)``
    and ``denom = max(|commanded|, delta)`` are both already invariant. Factored
    out as a pure function (rather than left inline in
    ``SchaftUnifiedJoystickV1._unified_components``) so training, the
    deterministic evaluator and this module's own tests all exercise the
    identical formula -- the same reason ``command_region_weights`` above is a
    free function and not inlined either.
    """
    e = measured - commanded
    gate = (jp.abs(commanded) > delta).astype(jp.float32)
    denom = jp.maximum(jp.abs(commanded), delta)
    signed = measured * jp.sign(commanded)
    track = 1.0 / (1.0 + (e / sigma) ** 2)
    prog = gate * jp.clip(signed / denom, 0.0, 1.0)
    wrong = gate * jp.clip(-signed / denom, 0.0, 1.0)
    return DenseAxisTerms(track, prog, wrong)


def init_command_state(
    rng: jax.Array,
    spec: SchedulerSpec,
    compat: bool,
    canonical_start_side: bool,
    reset_yaw: jax.Array,
    root_xy: jax.Array,
) -> CommandState:
    """Build the reset command state.

    The gate takes its steady-state value immediately with no debounce delay and
    ``ticks_since_request_change = TRANSITION_TICKS`` (§8.5).
    """
    k_v, k_cat, k_cmd, k_hold, k_quiet, k_side = jax.random.split(rng, 6)
    v_world = _uniform(k_v, spec.compat_v_world[0], spec.compat_v_world[1])
    if compat:
        command = jp.array(
            [v_world * jp.cos(reset_yaw), -v_world * jp.sin(reset_yaw), 0.0]
        )
        category = jp.asarray(ucfg.CAT_FORWARD, dtype=jp.int32)
        heading_target = jp.asarray(0.0, dtype=jp.float32)
        hold = jp.asarray(1000000000, dtype=jp.int32)
    else:
        start_quiet = jax.random.bernoulli(k_quiet, spec.reset_quiet_probability)
        sampled_cat = sample_category(k_cat, spec)
        category = jp.where(
            start_quiet, jp.asarray(ucfg.CAT_QUIET, dtype=jp.int32), sampled_cat
        ).astype(jp.int32)
        command = sample_command_for_category(k_cmd, category, spec)
        heading_target = wrap_to_pi(reset_yaw)
        is_quiet = category == ucfg.CAT_QUIET
        is_yaw = category == ucfg.CAT_YAW
        hold = jp.where(
            is_quiet,
            _sample_hold(k_hold, spec.quiet_hold),
            jp.where(
                is_yaw,
                _sample_hold(k_hold, spec.yaw_hold),
                _sample_hold(k_hold, spec.moving_hold),
            ),
        ).astype(jp.int32)
    gate = jp.where(quiet_command(command), 0.0, 1.0).astype(jp.float32)
    raw_request = gate.astype(jp.int32)
    if canonical_start_side:
        next_side = _random_sign(k_side)
        start_side = jp.asarray(0.0, dtype=jp.float32)
    else:
        next_side = jp.asarray(0.0, dtype=jp.float32)
        start_side = jp.asarray(0.0, dtype=jp.float32)
    return CommandState(
        command=command.astype(jp.float32),
        command_target=command.astype(jp.float32),
        category=category,
        hold_ticks_remaining=hold,
        instant=jp.asarray(0, dtype=jp.int32),
        pending_command=jp.zeros(3, dtype=jp.float32),
        pending_category=jp.asarray(ucfg.CAT_QUIET, dtype=jp.int32),
        pending_hold=jp.asarray(0, dtype=jp.int32),
        pending_valid=jp.asarray(0, dtype=jp.int32),
        raw_request=raw_request,
        raw_streak=jp.asarray(ucfg.DEBOUNCE_TICKS, dtype=jp.int32),
        locomotion_request=gate,
        ticks_since_request_change=jp.asarray(ucfg.TRANSITION_TICKS, dtype=jp.int32),
        heading_target=heading_target.astype(jp.float32),
        p_ref=root_xy.astype(jp.float32),
        v_world=v_world.astype(jp.float32),
        start_swing_side=start_side,
        next_start_swing_side=next_side,
    )


def _advance_scheduler(
    state: CommandState, rng: jax.Array, spec: SchedulerSpec
) -> CommandState:
    """Advance the shipped category scheduler by one tick (§8.4)."""
    k_switch, k_cat, k_cmd, k_hold, k_inst = jax.random.split(rng, 5)
    remaining = state.hold_ticks_remaining - 1
    expired = remaining <= 0
    resume = jp.logical_and(expired, state.pending_valid == 1)
    switch = jax.random.bernoulli(k_switch, spec.category_switch_probability)
    fresh_cat = jp.where(switch, sample_category(k_cat, spec), state.category).astype(
        jp.int32
    )
    fresh_cmd = sample_command_for_category(k_cmd, fresh_cat, spec)
    reverses = _sign_reverses(state.command_target, fresh_cmd)
    quiet_hold = _sample_hold(k_hold, spec.quiet_hold)
    moving_hold = _sample_hold(k_hold, spec.moving_hold)
    yaw_hold = _sample_hold(k_hold, spec.yaw_hold)
    fresh_hold = jp.where(
        fresh_cat == ucfg.CAT_QUIET,
        quiet_hold,
        jp.where(fresh_cat == ucfg.CAT_YAW, yaw_hold, moving_hold),
    )
    route_target = jp.where(reverses, jp.zeros(3), fresh_cmd)
    route_cat = jp.where(
        reverses, jp.asarray(ucfg.CAT_QUIET, dtype=jp.int32), fresh_cat
    ).astype(jp.int32)
    route_hold = jp.where(reverses, quiet_hold, fresh_hold).astype(jp.int32)
    instant = jp.logical_and(
        jax.random.bernoulli(k_inst, spec.instant_transition_prob),
        jp.logical_not(reverses),
    ).astype(jp.int32)
    new_target = jp.where(resume, state.pending_command, route_target)
    new_cat = jp.where(resume, state.pending_category, route_cat).astype(jp.int32)
    new_hold = jp.where(resume, state.pending_hold, route_hold).astype(jp.int32)
    new_instant = jp.where(resume, jp.asarray(0, jp.int32), instant)
    store_pending = jp.logical_and(
        expired, jp.logical_and(reverses, jp.logical_not(resume))
    )
    pending_command = jp.where(store_pending, fresh_cmd, state.pending_command)
    pending_category = jp.where(
        store_pending, fresh_cat, state.pending_category
    ).astype(jp.int32)
    pending_hold = jp.where(store_pending, fresh_hold, state.pending_hold).astype(
        jp.int32
    )
    pending_valid = jp.where(
        store_pending, 1, jp.where(resume, 0, state.pending_valid)
    ).astype(jp.int32)
    command_target = jp.where(expired, new_target, state.command_target)
    category = jp.where(expired, new_cat, state.category).astype(jp.int32)
    hold_ticks = jp.where(expired, new_hold, remaining).astype(jp.int32)
    instant_flag = jp.where(expired, new_instant, state.instant).astype(jp.int32)
    delta = command_target - state.command
    stepped = state.command + jp.clip(delta, -spec.slew, spec.slew)
    command = jp.where(instant_flag == 1, command_target, stepped)
    command = jp.where(
        jp.abs(command_target - command) < 1e-09, command_target, command
    )
    return state._replace(
        command=command.astype(jp.float32),
        command_target=command_target.astype(jp.float32),
        category=category,
        hold_ticks_remaining=hold_ticks,
        instant=instant_flag,
        pending_command=pending_command.astype(jp.float32),
        pending_category=pending_category,
        pending_hold=pending_hold,
        pending_valid=pending_valid,
    )


_FROZEN_HOLD = 1000000000


def override_command(
    state: CommandState, active: jax.Array, forced_command: jax.Array
) -> CommandState:
    """JAX-traceable command override, shared by ``step_command_state`` and
    ``SchaftUnifiedJoystickV1.set_command`` (the latter for an immediate,
    between-tick observation rebuild)."""
    cmd = jp.where(active, forced_command, state.command)
    hold = jp.where(
        active, jp.asarray(_FROZEN_HOLD, dtype=jp.int32), state.hold_ticks_remaining
    )
    return state._replace(
        command=cmd.astype(jp.float32),
        command_target=cmd.astype(jp.float32),
        hold_ticks_remaining=hold,
    )


def step_command_state(
    state: CommandState,
    rng: jax.Array,
    spec: SchedulerSpec,
    compat: bool,
    yaw: jax.Array,
    ctrl_dt: float,
    force_active: jax.Array | None = None,
    forced_command: jax.Array | None = None,
) -> CommandState:
    """Advance command, gate, heading and path reference by one control tick.

    Args:
      state: previous state.
      rng: per-tick key (unused in compat mode).
      spec: scheduler parameters.
      compat: ``True`` for the U0/U1 compatibility generator (§8.1).
      yaw: measured forward yaw ``psi``, used only by the compat projection.
      ctrl_dt: control timestep.
      force_active: when not ``None``, a JAX bool/float array selecting -- via
        ``jp.where``, fully traceable -- whether ``forced_command`` overrides
        the generator's output this tick. Used by the deterministic
        evaluator/renderer to hold a fixed shipped-mode command across many
        ticks without touching the stochastic scheduler; always inactive
        (zeros) during training so this is a no-op there. Applied BEFORE the
        gate and heading-integration steps, so both correctly reflect the
        forced command (the same equations the scheduler path uses, §10 of the
        implementation brief: "training and evaluator must use the same
        command and reference equations").
      forced_command: the (3,) body-frame command to hold when
        ``force_active`` is true. Ignored if ``force_active`` is ``None``.

    Returns:
      The updated :class:`CommandState`. ``p_ref`` is *not* integrated here; see
      :func:`integrate_path_reference`, which needs the post-step root position.
    """
    if compat:
        cmd = jp.array(
            [state.v_world * jp.cos(yaw), -state.v_world * jp.sin(yaw), 0.0],
            dtype=jp.float32,
        )
        generated = state._replace(command=cmd, command_target=cmd)
    else:
        generated = _advance_scheduler(state, rng, spec)
    if force_active is not None:
        generated = override_command(generated, force_active, forced_command)
    new = update_gate(generated, generated.command)
    if compat:
        heading_target = jp.asarray(0.0, dtype=jp.float32)
    else:
        heading_target = wrap_to_pi(new.heading_target + new.command[2] * ctrl_dt)
    return new._replace(heading_target=heading_target.astype(jp.float32))


def integrate_path_reference(
    state: CommandState, compat: bool, ctrl_dt: float
) -> CommandState:
    """Integrate ``p_ref`` one tick using the stage's ``v_ref_world`` (§9.4)."""
    v_ref = reference_velocity_world(state, compat)
    return state._replace(p_ref=(state.p_ref + v_ref * ctrl_dt).astype(jp.float32))


def begin_locomotion_bout(state: CommandState) -> CommandState:
    """Apply the §9.3 canonical initiation-side rule on a QUIET->MOVING edge.

    ``start_swing_side`` takes the pre-drawn ``next_start_swing_side``; the
    successor is then negated so repeated starts alternate.
    """
    return state._replace(
        start_swing_side=state.next_start_swing_side,
        next_start_swing_side=-state.next_start_swing_side,
    )


_INFO_PREFIX = "ucmd_"
_FIELDS = CommandState._fields


def to_info(state: CommandState) -> dict:
    """Flatten a :class:`CommandState` into ``state.info`` entries."""
    return {f"{_INFO_PREFIX}{name}": getattr(state, name) for name in _FIELDS}


def from_info(info: dict) -> CommandState:
    """Rebuild a :class:`CommandState` from ``state.info``."""
    try:
        return CommandState(**{name: info[f"{_INFO_PREFIX}{name}"] for name in _FIELDS})
    except KeyError as exc:
        raise KeyError(
            f"info dict is missing unified command-state key {exc}; the env was not reset by SchaftUnifiedJoystickV1.reset"
        ) from exc


def has_command_state(info: dict) -> bool:
    """True when ``info`` already carries the unified command state."""
    return f"{_INFO_PREFIX}command" in info


def neutral_command_state() -> CommandState:
    """An all-zero state used only while an ancestor ``reset`` is still running.

    ``ResidualForwardWalking*.reset`` calls ``self._get_obs`` several times
    before :meth:`SchaftUnifiedJoystickV1.reset` has installed the real command
    state. Those intermediate observations are discarded — the unified ``reset``
    rebuilds the observation from the real state before returning — so a defined
    neutral value here is correct and keeps ``_get_obs`` total.
    """
    return CommandState(
        command=jp.zeros(3, dtype=jp.float32),
        command_target=jp.zeros(3, dtype=jp.float32),
        category=jp.asarray(ucfg.CAT_QUIET, dtype=jp.int32),
        hold_ticks_remaining=jp.asarray(0, dtype=jp.int32),
        instant=jp.asarray(0, dtype=jp.int32),
        pending_command=jp.zeros(3, dtype=jp.float32),
        pending_category=jp.asarray(ucfg.CAT_QUIET, dtype=jp.int32),
        pending_hold=jp.asarray(0, dtype=jp.int32),
        pending_valid=jp.asarray(0, dtype=jp.int32),
        raw_request=jp.asarray(0, dtype=jp.int32),
        raw_streak=jp.asarray(0, dtype=jp.int32),
        locomotion_request=jp.asarray(0.0, dtype=jp.float32),
        ticks_since_request_change=jp.asarray(0, dtype=jp.int32),
        heading_target=jp.asarray(0.0, dtype=jp.float32),
        p_ref=jp.zeros(2, dtype=jp.float32),
        v_world=jp.asarray(0.0, dtype=jp.float32),
        start_swing_side=jp.asarray(0.0, dtype=jp.float32),
        next_start_swing_side=jp.asarray(0.0, dtype=jp.float32),
    )


U4_MODE_NORMAL: int = 0
U4_MODE_FIXED_YAW_HOLD: int = 1
U4_MODE_DISCRETE_YAW: int = 2
U4_MODE_YAW_TRANSITIONS: int = 3
U4_MODE_CURVED_WALK: int = 4
NUM_U4_MODES: int = 5
_DISCRETE_YAW_MAGNITUDES = (0.08, 0.14, 0.2)
_YAW_TRANSITIONS_LOW = 0.08
_YAW_TRANSITIONS_HIGH = 0.2
_YAW_TRANSITIONS_VX = 0.08


def sample_u4_unified_episode(
    rng: jax.Array, mode_probs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Draws one episode's ``(mode_id, params)`` for U4_UNIFIED_YAW_FINAL.

    ``mode_id``: ``()`` i32 in ``{0..4}`` -- see the ``U4_MODE_*`` constants.

    ``params``: ``(8,)`` f32, meaning dependent on ``mode_id``, consumed only
    by :func:`u4_unified_command`:

      0: sign (+-1); used by modes 1, 2, 4 and by YAW_TRANSITIONS patterns 1/4/5.
      1: mode 2 (DISCRETE_YAW) magnitude in {0.08,0.14,0.20}; mode 4
         (CURVED_WALK) vx in [0.05, 0.12].
      2: mode 2 vx in [0.06, 0.10]; mode 4 radius in [0.35, 1.20].
      3: randomized hold-tick count for modes 1/2/4 (ranges differ per mode;
         see the per-mode sampling below).
      4: reserved, always 0.
      5: YAW_TRANSITIONS is_ramp flag (1.0 = linear ramp, 0.0 = step).
      6: YAW_TRANSITIONS ramp duration in ticks, in [15, 30].
      7: YAW_TRANSITIONS pattern id in {1..5}.

    All branches are computed unconditionally (required for `jax.jit`); only
    the branch matching the sampled ``mode_id`` is ever read downstream.
    """
    (
        k_mode,
        k_sign,
        k_mag,
        k_vx,
        k_radius,
        k_hold,
        k_ramp_flag,
        k_ramp_ticks,
        k_pattern,
    ) = jax.random.split(rng, 9)
    logits = jp.log(jp.maximum(mode_probs, 0.0))
    mode_id = jax.random.categorical(k_mode, logits).astype(jp.int32)
    sign = _random_sign(k_sign)
    discrete_mag = jax.random.choice(
        k_mag, jp.asarray(_DISCRETE_YAW_MAGNITUDES, dtype=jp.float32)
    )
    discrete_vx = _uniform(k_vx, 0.06, 0.1)
    curved_vx = _uniform(k_vx, 0.05, 0.12)
    radius = _uniform(k_radius, 0.35, 1.2)
    fixed_hold = _sample_hold(k_hold, jp.asarray([600, 1000], dtype=jp.int32))
    discrete_hold = _sample_hold(k_hold, jp.asarray([250, 600], dtype=jp.int32))
    curved_hold = _sample_hold(k_hold, jp.asarray([200, 700], dtype=jp.int32))
    hold = jp.where(
        mode_id == U4_MODE_FIXED_YAW_HOLD,
        fixed_hold.astype(jp.float32),
        jp.where(
            mode_id == U4_MODE_DISCRETE_YAW,
            discrete_hold.astype(jp.float32),
            curved_hold.astype(jp.float32),
        ),
    )
    is_ramp = jax.random.bernoulli(k_ramp_flag, 0.5).astype(jp.float32)
    ramp_ticks = _sample_hold(
        k_ramp_ticks, jp.asarray([15, 30], dtype=jp.int32)
    ).astype(jp.float32)
    pattern_id = jax.random.randint(k_pattern, (), 1, 6).astype(jp.float32)
    params = jp.stack(
        [
            sign,
            jp.where(mode_id == U4_MODE_DISCRETE_YAW, discrete_mag, curved_vx),
            jp.where(mode_id == U4_MODE_DISCRETE_YAW, discrete_vx, radius),
            hold,
            jp.zeros(()),
            is_ramp,
            ramp_ticks,
            pattern_id,
        ]
    ).astype(jp.float32)
    return (mode_id, params)


def _blend(
    t: jax.Array,
    boundary: jax.Array,
    ramp_ticks: jax.Array,
    before: jax.Array,
    after: jax.Array,
    is_ramp: jax.Array,
) -> jax.Array:
    """Step or linear-ramp transition from ``before`` to ``after`` at ``boundary``.

    The ramp spans ``[boundary - ramp_ticks/2, boundary + ramp_ticks/2]``; a
    step is the ``ramp_ticks -> 0`` limit of the same formula, computed
    directly (not as a limit) to stay exact when ``is_ramp`` is 0.
    """
    step_val = jp.where(t < boundary, before, after)
    half = ramp_ticks / 2.0
    frac = jp.clip((t - (boundary - half)) / jp.maximum(ramp_ticks, 1e-06), 0.0, 1.0)
    ramp_val = before + (after - before) * frac
    return jp.where(is_ramp > 0.5, ramp_val, step_val)


def _yaw_transitions_wz(
    sign: jax.Array,
    is_ramp: jax.Array,
    ramp_ticks: jax.Array,
    pattern_id: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """The five YAW_TRANSITIONS patterns' commanded wz at tick ``t``.

    1: straight -> yaw -> straight (boundaries at 300/700 ticks).
    2: +yaw -> straight -> -yaw (300/700).
    3: -yaw -> straight -> +yaw (300/700, the exact mirror of 2).
    4: yaw magnitude increase, low -> high (boundary at 500).
    5: yaw magnitude decrease, high -> low (boundary at 500, mirror of 4).

    Every 1000-tick episode therefore has segments of >= 300 ticks either
    side of any transition, comfortably longer than the 15-30 tick ramp.
    """
    m, lo = (_YAW_TRANSITIONS_HIGH, _YAW_TRANSITIONS_LOW)

    def _pick(x1, x2, x3, x4, x5):
        return jp.where(
            pattern_id < 1.5,
            x1,
            jp.where(
                pattern_id < 2.5,
                x2,
                jp.where(pattern_id < 3.5, x3, jp.where(pattern_id < 4.5, x4, x5)),
            ),
        )

    v0 = _pick(0.0, m, -m, sign * lo, sign * m)
    v1 = _pick(sign * m, 0.0, 0.0, sign * m, sign * lo)
    v2 = _pick(0.0, -m, m, sign * m, sign * lo)
    b1 = _pick(300.0, 300.0, 300.0, 500.0, 500.0)
    b2 = _pick(700.0, 700.0, 700.0, 1000.0, 1000.0)
    stage1 = _blend(t, b1, ramp_ticks, v0, v1, is_ramp)
    return _blend(t, b2, ramp_ticks, stage1, v2, is_ramp)


def unwrap_heading_step(
    prev_unwrapped: jax.Array, prev_raw_yaw: jax.Array, raw_yaw: jax.Array
) -> jax.Array:
    """One tick of the numerically-safe unwrapped heading accumulator:

    ``yaw_unwrapped[t] = yaw_unwrapped[t-1] + wrap(yaw[t] - yaw[t-1])``.

    Never differences two wrapped angles directly (which would show a
    spurious ~2*pi jump at a +-pi crossing) -- the same technique
    `unified_command_eval.py`'s own ``heading_change`` accumulator already
    uses for its per-tick unwrapped total.
    """
    return prev_unwrapped + wrap_to_pi(raw_yaw - prev_raw_yaw)


def heading_response_ratio(
    command_wz: jax.Array,
    yaw_unwrapped: jax.Array,
    segment_start_yaw_unwrapped: jax.Array,
    segment_elapsed_time: jax.Array,
) -> jax.Array:
    """``actual_signed_heading / max(expected_heading, 0.10)`` (§ "Accumulated
    heading progress"). Positive for correct accumulated motion, ~0 for pure
    oscillation with no net turn, negative for accumulated wrong-direction
    turning. Callers gate this to ``segment_elapsed_time >= 0.50`` themselves
    (this function has no opinion on validity, only the formula)."""
    actual_signed_heading = jp.sign(command_wz) * (
        yaw_unwrapped - segment_start_yaw_unwrapped
    )
    expected_heading = jp.abs(command_wz) * segment_elapsed_time
    return actual_signed_heading / jp.maximum(expected_heading, 0.1)


def wz_dc_alpha(ctrl_dt: float, tau: jax.Array) -> jax.Array:
    """``alpha = 1 - exp(-ctrl_dt / tau)`` -- the low-pass update weight."""
    return 1.0 - jp.exp(-ctrl_dt / tau)


def update_wz_dc(
    wz_dc_prev: jax.Array, raw_wz: jax.Array, alpha: jax.Array
) -> jax.Array:
    """One tick of ``wz_dc[t] = (1-alpha)*wz_dc[t-1] + alpha*raw_wz[t]``."""
    return (1.0 - alpha) * wz_dc_prev + alpha * raw_wz


STRICT_YAW_HOLD_VX: float = 0.08
STRICT_YAW_HOLD_START_TICK: int = 100
STRICT_YAW_HOLD_END_TICK: int = 900
STRICT_YAW_HOLD_MAGNITUDE: float = 0.2
STRICT_YAW_ACTIVE_TICKS: int = STRICT_YAW_HOLD_END_TICK - STRICT_YAW_HOLD_START_TICK
STRICT_YAW_HEADING_VALID_TICKS: int = STRICT_YAW_ACTIVE_TICKS - 24
STRICT_YAW_POST_TURN_WINDOW_TICKS: int = 50


def sample_strict_yaw_episode(
    rng: jax.Array, probability: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Draws one episode's ``(active, sign)`` for STRICT_YAW_HOLD.

    ``active``: ``()`` f32 in ``{0, 1}``. ``probability=0.0`` (every stage
    except U4_STRICT_SIGNED_YAW_DC_ACQUISITION) makes ``active`` always 0.
    ``sign``: ``()`` f32 in ``{-1, +1}``, balanced 50/50, independent of
    ``active`` (drawn from a separate subkey either way, so its marginal
    distribution never depends on whether this episode is active).
    """
    k_active, k_sign = jax.random.split(rng, 2)
    active = jax.random.bernoulli(k_active, probability).astype(jp.float32)
    sign = _random_sign(k_sign)
    return (active, sign)


def strict_yaw_hold_command(
    active: jax.Array, sign: jax.Array, step_count: jax.Array
) -> jax.Array:
    """The scripted ``[vx, vy, wz]`` command at ``step_count``.

    ``active=0`` -> the zero vector (the override is inactive then anyway;
    this is a well-formed placeholder, never read).
    """
    t = step_count
    in_yaw_window = jp.logical_and(
        t >= STRICT_YAW_HOLD_START_TICK, t < STRICT_YAW_HOLD_END_TICK
    )
    wz = jp.where(in_yaw_window, sign * STRICT_YAW_HOLD_MAGNITUDE, 0.0)
    cmd = jp.array([STRICT_YAW_HOLD_VX, 0.0, wz], dtype=jp.float32)
    return jp.where(active > 0.5, cmd, jp.zeros(3, dtype=jp.float32))


def u4_unified_command(
    mode_id: jax.Array, params: jax.Array, step_count: jax.Array
) -> jax.Array:
    """The scripted ``[vx, vy, wz]`` command at ``step_count`` for ``mode_id``.

    Pure function of ``(mode_id, params, step_count)`` -- no RNG, no hidden
    state -- so it is independently testable and, fed through
    `step_command_state`'s `force_active`/`forced_command`, reproduces
    bit-identically whatever a caller computes from the same inputs.
    """
    sign, a, b, hold = (params[0], params[1], params[2], params[3])
    is_ramp, ramp_ticks, pattern_id = (params[5], params[6], params[7])
    t = step_count.astype(jp.float32)
    fixed_cmd = jp.array([0.08, 0.0, jp.where(t < hold, sign * 0.2, 0.0)])
    discrete_cmd = jp.array([b, 0.0, jp.where(t < hold, sign * a, 0.0)])
    curved_wz_mag = jp.clip(a / jp.maximum(b, 1e-06), 0.05, 0.25)
    curved_cmd = jp.array([a, 0.0, jp.where(t < hold, sign * curved_wz_mag, 0.0)])
    transitions_cmd = jp.array(
        [
            _YAW_TRANSITIONS_VX,
            0.0,
            _yaw_transitions_wz(sign, is_ramp, ramp_ticks, pattern_id, t),
        ]
    )
    cmd = jp.where(
        mode_id == U4_MODE_FIXED_YAW_HOLD,
        fixed_cmd,
        jp.where(
            mode_id == U4_MODE_DISCRETE_YAW,
            discrete_cmd,
            jp.where(
                mode_id == U4_MODE_YAW_TRANSITIONS,
                transitions_cmd,
                jp.where(mode_id == U4_MODE_CURVED_WALK, curved_cmd, jp.zeros(3)),
            ),
        ),
    )
    return cmd.astype(jp.float32)
