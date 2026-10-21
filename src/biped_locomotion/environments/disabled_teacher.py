"""U13-only inert teacher boundary. Assisted historical experiments are excluded."""

import jax.numpy as jp

N_GATE_STATES = 1


def gate_ramp_alpha(dt):
    return 0.0


def teacher_residual(*args):
    return jp.zeros(12), jp.zeros(N_GATE_STATES)
