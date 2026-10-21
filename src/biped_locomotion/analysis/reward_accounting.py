"""Separate raw terms, coefficient-weighted terms and dt-scaled return."""


def account(raw, coefficients, dt):
    weighted = {k: float(v) * float(coefficients[k]) for k, v in raw.items()}
    actual = {k: v * dt for k, v in weighted.items()}
    positive = sum(max(v, 0) for v in actual.values())
    return {
        "raw": dict(raw),
        "weighted": weighted,
        "actual": actual,
        "total": sum(actual.values()),
        "positive_mass_share": {
            k: max(v, 0) / positive if positive else 0 for k, v in actual.items()
        },
    }
