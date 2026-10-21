"""V24 placement terms, using the original validity and event gating."""


def bilateral_terms(
    left,
    right,
    pair_valid,
    previous_abs_imbalance,
    completed,
    previous_pair_valid,
    path_error_y,
):
    b = pair_valid * (right - left) / (right + left + 1e-6)
    return {
        "bilateral_symmetry": pair_valid * abs(right - left) / 0.10,
        "imbalance_growth": completed
        * pair_valid
        * previous_pair_valid
        * min(max(abs(b) - previous_abs_imbalance, 0) / 0.05, 1),
        "persistent_lateral_path": max(abs(path_error_y) - 0.14, 0) / 0.10,
    }
