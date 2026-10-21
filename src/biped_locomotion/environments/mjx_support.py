# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import jax
import mujoco
from mujoco import mjx
from flax import struct

Observation = Any


def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    naconmax: Optional[int] = None,
    naccdmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
    """Initialize MJX Data."""
    data = mjx.make_data(
        model,
        impl=impl,
        naconmax=naconmax,
        naccdmax=naccdmax,
        njmax=njmax,
        device=device,
    )
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if act is not None:
        data = data.replace(act=act)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
    return data


def step(
    model: mjx.Model, data: mjx.Data, action: jax.Array, n_substeps: int = 1
) -> mjx.Data:
    def single_step(data, _):
        data = data.replace(ctrl=action)
        data = mjx.step(model, data)
        return (data, None)

    return jax.lax.scan(single_step, data, (), n_substeps)[0]


@struct.dataclass
class State:
    """Environment state for training and inference."""

    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]

    def tree_replace(
        self, params: Dict[str, Optional[jax.typing.ArrayLike]]
    ) -> "State":
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new


def _tree_replace(
    base: Any, attr: Sequence[str], val: Optional[jax.typing.ArrayLike]
) -> Any:
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        raise NotImplementedError("List attributes are not supported.")
    if len(attr) == 1:
        return base.replace(**{attr[0]: val})
    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]
