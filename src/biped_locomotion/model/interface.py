"""External bindings only. No robot description or model-specific defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Any
import json


@dataclass(frozen=True)
class ModelSpec:
    model_path: Path
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    roles: Mapping[str, str]
    contact_geoms: Mapping[str, str]
    contact_bodies: Mapping[str, str]
    floor_geom: str
    root_body: str
    sensors: Mapping[str, str]
    reference_qpos: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort_limits: tuple[float, ...]
    action_scale: tuple[float, ...]
    environment_metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_pose_mapping: Mapping[str, float] = field(default_factory=dict)

    @property
    def foot_geoms(self):
        return (self.contact_geoms["left"], self.contact_geoms["right"])

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        d = json.loads(path.read_text())
        model = Path(d["model_path"])
        d["model_path"] = model if model.is_absolute() else path.parent / model
        for key in (
            "joint_names",
            "actuator_names",
            "reference_qpos",
            "kp",
            "kd",
            "effort_limits",
            "action_scale",
        ):
            d[key] = tuple(d[key])
        spec = cls(**d)
        from .validation import validate_spec

        validate_spec(spec)
        return spec


class ModelInterface(Protocol):
    spec: ModelSpec

    def reset(self, rng): ...
    def step(self, state, action): ...
