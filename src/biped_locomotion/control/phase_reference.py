"""Runtime supplied reference and phase envelope; no nominal robot pose."""

import numpy as np


def phase_reference(nominal, offset, gate):
    """General phase-conditioned reference. V24's active offset is zero."""
    if not 0 <= gate <= 1:
        raise ValueError("phase gate must lie in [0,1]")
    nominal, offset = np.asarray(nominal), np.asarray(offset)
    if nominal.shape != offset.shape:
        raise ValueError("reference shape mismatch")
    return nominal + gate * offset
