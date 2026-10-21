"""Short/public scenarios and historical horizons; no automatic rollout."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    command: tuple[float, float, float]
    horizon_s: float
    deterministic: bool = True


V24_CANONICAL = Scenario("historical_forward", (0.10, 0.0, 0.0), 60.0)
U13_FORWARD = Scenario("historical_u13_forward", (0.08, 0.0, 0.0), 20.0)
