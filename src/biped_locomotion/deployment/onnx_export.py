"""Export of the deterministic PPO actor to ONNX, with a NumPy reference forward pass.

Code only: no checkpoint or trained weights are distributed with this repository. The
functions take plain NumPy arrays, so they can be tested with synthetic weights; loading a
Brax PPO checkpoint (:func:`load_brax_checkpoint`) requires an authorized checkpoint that the
user supplies externally.

Exported graph (``obs`` ``[N, 57]`` -> ``continuous_actions`` ``[N, 12]``)::

    (obs - mean) / std -> [MatMul + Add + SiLU]* -> MatMul + Add -> split(12, 12) -> tanh(first 12)

i.e. observation normalization is embedded, the head emits 12 Gaussian locations + 12 scale
logits, and the deterministic action is ``tanh(location)`` (see docs/action_space.md).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OBS_SIZE = 57
ACTION_SIZE = 12
HEAD_SIZE = 2 * ACTION_SIZE
SILU_NAMES = {"silu", "swish"}  # the graph hard-codes x * sigmoid(x)


class ExportError(RuntimeError):
    """Raised when checkpoint contents do not match the expected actor contract."""


Layer = tuple[np.ndarray, np.ndarray]  # (kernel [in, out], bias [out])


def _as_float32_array(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        raise ExportError(f"{name} dtype {arr.dtype} != float32")
    if not np.all(np.isfinite(arr)):
        raise ExportError(f"{name} contains non-finite values")
    return arr


def validate_actor(mean: np.ndarray, std: np.ndarray, layers: Sequence[Layer]) -> None:
    """Check normalizer and layer shapes form a ``57 -> ... -> 24`` MLP."""
    for name, arr in (("mean", mean), ("std", std)):
        if arr.shape != (OBS_SIZE,):
            raise ExportError(f"normalizer {name} shape {arr.shape} != ({OBS_SIZE},)")
    if not np.all(std > 0.0):
        bad = np.nonzero(std <= 0.0)[0].tolist()
        raise ExportError(f"normalizer std has non-positive entries at indices {bad}")
    if len(layers) < 2:
        raise ExportError("actor needs at least one hidden layer and an output layer")
    width = OBS_SIZE
    for i, (kernel, bias) in enumerate(layers):
        if kernel.ndim != 2 or kernel.shape[0] != width:
            raise ExportError(f"layer {i} kernel shape {kernel.shape} does not accept width {width}")
        if bias.shape != (kernel.shape[1],):
            raise ExportError(f"layer {i} bias shape {bias.shape} != ({kernel.shape[1]},)")
        width = kernel.shape[1]
    if width != HEAD_SIZE:
        raise ExportError(f"final layer width {width} != {HEAD_SIZE} (12 locations + 12 scales)")


def reference_forward(obs: np.ndarray, mean: np.ndarray, std: np.ndarray, layers: Sequence[Layer]) -> np.ndarray:
    """NumPy float32 reference of the exported graph (deterministic actions)."""
    x = ((obs.astype(np.float32) - mean) / std).astype(np.float32)
    for kernel, bias in layers[:-1]:
        h = x @ kernel + bias
        x = (h / (1.0 + np.exp(-h))).astype(np.float32)  # SiLU
    kernel, bias = layers[-1]
    logits = x @ kernel + bias
    return np.tanh(logits[:, :ACTION_SIZE]).astype(np.float32)


def build_onnx_model(mean: np.ndarray, std: np.ndarray, layers: Sequence[Layer]):
    """Return an ``onnx.ModelProto`` for the deterministic actor (embedded normalization)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    validate_actor(mean, std, layers)
    initializers = [
        numpy_helper.from_array(mean.reshape(1, OBS_SIZE), name="obs_mean"),
        numpy_helper.from_array(std.reshape(1, OBS_SIZE), name="obs_std"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["obs_centered"], name="normalize_sub"),
        helper.make_node("Div", ["obs_centered", "obs_std"], ["h0_in"], name="normalize_div"),
    ]
    current = "h0_in"
    for i, (kernel, bias) in enumerate(layers):
        initializers.append(numpy_helper.from_array(kernel, name=f"w{i}"))
        initializers.append(numpy_helper.from_array(bias, name=f"b{i}"))
        last = i == len(layers) - 1
        dense_out = "logits" if last else f"dense{i}"
        nodes.append(helper.make_node("MatMul", [current, f"w{i}"], [f"mm{i}"], name=f"matmul_{i}"))
        nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [dense_out], name=f"add_{i}"))
        if not last:
            nodes.append(helper.make_node("Sigmoid", [dense_out], [f"sig{i}"], name=f"sigmoid_{i}"))
            nodes.append(helper.make_node("Mul", [dense_out, f"sig{i}"], [f"silu{i}"], name=f"silu_{i}"))
            current = f"silu{i}"
    initializers.append(numpy_helper.from_array(np.array([ACTION_SIZE, ACTION_SIZE], dtype=np.int64), name="split_sizes"))
    nodes.append(helper.make_node("Split", ["logits", "split_sizes"], ["mean_logits", "unused_scale"], axis=1, name="split_mean_scale"))
    nodes.append(helper.make_node("Tanh", ["mean_logits"], ["continuous_actions"], name="tanh_mean"))

    graph = helper.make_graph(
        nodes,
        "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", OBS_SIZE])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, ["N", ACTION_SIZE])],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="biped_locomotion_onnx_export", opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = min(model.ir_version, 11)
    onnx.checker.check_model(model)
    return model


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_export(model, output_dir: Path, filename: str = "actor.onnx", extra: dict | None = None) -> Path:
    """Save the model and an ``EXPORT_MANIFEST.json`` whose layout manifest_verify.cpp reads."""
    import onnx

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / filename
    onnx.save(model, onnx_path)
    manifest = {
        "artifacts": {filename: {"sha256": sha256_file(onnx_path), "bytes": onnx_path.stat().st_size}},
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SIZE,
        "deterministic_action": "tanh(location)",
        "note": "Software export record. The checkpoint and trained weights are not part of this repository.",
    }
    if extra:
        manifest.update(extra)
    (output_dir / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return onnx_path


def make_synthetic_actor(seed: int = 0, hidden: Sequence[int] = (32, 16)):
    """Random-weight actor for software tests. These are NOT trained weights."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.2, OBS_SIZE).astype(np.float32)
    std = rng.uniform(0.5, 1.5, OBS_SIZE).astype(np.float32)
    dims = [OBS_SIZE, *hidden, HEAD_SIZE]
    layers = [
        (
            (rng.normal(0.0, 1.0 / np.sqrt(a), (a, b))).astype(np.float32),
            rng.normal(0.0, 0.05, b).astype(np.float32),
        )
        for a, b in zip(dims[:-1], dims[1:])
    ]
    return mean, std, layers


def extract_normalizer(normalizer_params: Any) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std from Brax ``RunningStatisticsState`` (optionally a ``{"state": ...}`` dict)."""
    mean, std = normalizer_params.mean, normalizer_params.std
    if isinstance(mean, dict):
        if "state" not in mean:
            raise ExportError("normalizer.mean has no 'state' leaf")
        mean, std = mean["state"], std["state"]
    return _as_float32_array(mean, "normalizer mean"), _as_float32_array(std, "normalizer std")


def extract_layers(policy_params: Any) -> list[Layer]:
    """Dense layers ``hidden_0 .. hidden_{n-1}`` from Brax policy params, in order."""
    params = policy_params["params"]
    count = len(params)
    expected = [f"hidden_{i}" for i in range(count)]
    if set(params.keys()) != set(expected):
        raise ExportError(f"policy layer keys {sorted(params.keys())} != {expected}")
    return [
        (_as_float32_array(params[k]["kernel"], f"{k}.kernel"), _as_float32_array(params[k]["bias"], f"{k}.bias"))
        for k in expected
    ]


def check_activation(checkpoint_dir: Path) -> None:
    """Reject checkpoints whose activation is not SiLU, if the network config records it."""
    cfg_path = checkpoint_dir / "ppo_network_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    activation = str(cfg.get("network_factory_kwargs", {}).get("activation", "silu")).lower()
    if activation not in SILU_NAMES:
        raise ExportError(f"activation '{activation}' is not supported (graph hard-codes SiLU)")


def load_brax_checkpoint(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Layer]]:
    """Load an authorized Brax PPO checkpoint (requires the ``simulation`` extra)."""
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    check_activation(checkpoint_dir)
    normalizer, policy_params, _ = ppo_checkpoint.load(str(checkpoint_dir))
    mean, std = extract_normalizer(normalizer)
    layers = [(np.asarray(k), np.asarray(b)) for k, b in extract_layers(policy_params)]
    validate_actor(mean, std, layers)
    return mean, std, layers
