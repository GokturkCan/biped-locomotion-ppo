# Adapted for the modular research release; behavior changes are documented in the extraction manifest.
# Copyright 2025 DeepMind Technologies Limited
# Licensed under Apache-2.0; see LICENSE.
from __future__ import annotations

"""Qualified events are produced by the extracted acquisition kernels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GaitEvents:
    liftoff_left: int = 0
    liftoff_right: int = 0
    touchdown_left: int = 0
    touchdown_right: int = 0
    support_transfer: int = 0
    same_side_repeat: int = 0
    unexpected_side: int = 0
    transfer_timeout: int = 0


import numpy as jp


def qualify_foot(
    config,
    side: str,
    contact: object,
    opposite_contact: object,
    height_gain: object,
    prev_contact: object,
    valid_ground_load_gate: object,
    info: dict[str, object],
) -> dict[str, object]:
    contact_f = jp.float32(contact)
    no_contact_f = 1.0 - contact_f
    raw_liftoff = jp.float32((prev_contact > 0.5) & (contact_f < 0.5))
    raw_touchdown = jp.float32((prev_contact < 0.5) & (contact_f > 0.5))
    no_contact_streak = jp.where(
        contact_f > 0.5, 0.0, info[f"v8_consecutive_no_contact_steps_{side}"] + 1.0
    )
    previous_max = info[f"v8_max_height_gain_during_air_{side}"]
    prev_air_qualified = info[f"v8_air_qualified_{side}"]
    max_air_height = jp.where(
        contact_f < 0.5,
        jp.maximum(previous_max, height_gain),
        jp.where(prev_air_qualified > 0.5, previous_max, 0.0),
    )
    contact_streak = jp.where(
        contact_f > 0.5, info[f"v8_consecutive_contact_steps_{side}"] + 1.0, 0.0
    )
    qualified_liftoff = (
        jp.float32(prev_air_qualified < 0.5)
        * jp.float32(no_contact_streak >= config.v8_liftoff_no_contact_steps)
        * jp.float32(max_air_height >= config.v8_liftoff_height_gain)
        * jp.float32(opposite_contact)
        * jp.float32(valid_ground_load_gate > 0.999)
    )
    air_qualified = jp.maximum(prev_air_qualified, qualified_liftoff)
    steps_since = info[f"v8_steps_since_last_qualified_touchdown_{side}"]
    qualified_touchdown = (
        air_qualified
        * jp.float32(contact_streak == config.v8_touchdown_contact_confirm_steps)
        * jp.float32(steps_since >= config.v8_min_steps_between_qualified_touchdowns)
    )
    next_air_qualified = jp.where(qualified_touchdown > 0.5, 0.0, air_qualified)
    next_max_air_height = jp.where(qualified_touchdown > 0.5, 0.0, max_air_height)
    next_steps_since = jp.where(qualified_touchdown > 0.5, 0.0, steps_since + 1.0)
    return {
        "raw_liftoff": raw_liftoff,
        "raw_touchdown": raw_touchdown,
        "qualified_liftoff": qualified_liftoff,
        "qualified_touchdown": qualified_touchdown,
        "no_contact_streak": no_contact_streak,
        "contact_streak": contact_streak,
        "max_air_height": max_air_height,
        "next_no_contact_streak": no_contact_streak,
        "next_max_air_height": next_max_air_height,
        "next_air_qualified": next_air_qualified,
        "next_contact_streak": contact_streak,
        "next_steps_since": next_steps_since,
    }
