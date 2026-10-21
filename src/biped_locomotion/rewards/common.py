"""Reward accounting for explicitly emitted runtime terms."""


def weighted_reward(terms, scales, dt):
    if dt <= 0:
        raise ValueError("dt must be positive")
    missing = set(terms) - set(scales)
    if missing:
        raise ValueError(f"missing coefficients: {sorted(missing)}")
    contributions = {k: float(v) * float(scales[k]) * dt for k, v in terms.items()}
    return max(-10000.0, min(10000.0, sum(contributions.values()))), contributions
