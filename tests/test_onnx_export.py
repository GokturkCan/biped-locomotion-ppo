"""Software-level tests of the ONNX export code, using synthetic random weights only."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from biped_locomotion.deployment import onnx_export as ox


def test_synthetic_actor_shapes_and_reference_range():
    mean, std, layers = ox.make_synthetic_actor(seed=1)
    ox.validate_actor(mean, std, layers)
    obs = np.random.default_rng(0).normal(size=(5, ox.OBS_SIZE)).astype(np.float32)
    out = ox.reference_forward(obs, mean, std, layers)
    assert out.shape == (5, ox.ACTION_SIZE) and out.dtype == np.float32
    assert np.all(np.abs(out) <= 1.0)  # tanh head


def test_validation_rejects_bad_contracts():
    mean, std, layers = ox.make_synthetic_actor(seed=0)
    with pytest.raises(ox.ExportError):
        ox.validate_actor(mean[:-1], std, layers)
    with pytest.raises(ox.ExportError):
        ox.validate_actor(mean, np.zeros_like(std), layers)
    with pytest.raises(ox.ExportError):
        ox.validate_actor(mean, std, layers[:1])  # no hidden layer
    kernel, bias = layers[-1]
    with pytest.raises(ox.ExportError):
        ox.validate_actor(mean, std, [*layers[:-1], (kernel[:, :12], bias[:12])])  # head must be 24 wide
    with pytest.raises(ox.ExportError):
        ox.validate_actor(mean, std, [(layers[0][0][:-1], layers[0][1]), *layers[1:]])


def test_extract_layers_orders_by_index_and_checks_dtype():
    _, _, layers = ox.make_synthetic_actor(seed=0)
    params = {"params": {f"hidden_{i}": {"kernel": k, "bias": b} for i, (k, b) in reversed(list(enumerate(layers)))}}
    got = ox.extract_layers(params)
    assert all(np.array_equal(g[0], l[0]) for g, l in zip(got, layers))
    params["params"]["hidden_0"]["kernel"] = layers[0][0].astype(np.float64)
    with pytest.raises(ox.ExportError):
        ox.extract_layers(params)
    with pytest.raises(ox.ExportError):
        ox.extract_layers({"params": {"hidden_0": {"kernel": layers[0][0], "bias": layers[0][1]}, "other": {}}})


def test_extract_normalizer_accepts_state_dict_and_rejects_nonfinite():
    mean = np.zeros(ox.OBS_SIZE, np.float32)
    std = np.ones(ox.OBS_SIZE, np.float32)
    m, s = ox.extract_normalizer(SimpleNamespace(mean={"state": mean}, std={"state": std}))
    assert m.shape == s.shape == (ox.OBS_SIZE,)
    bad = mean.copy()
    bad[0] = np.nan
    with pytest.raises(ox.ExportError):
        ox.extract_normalizer(SimpleNamespace(mean=bad, std=std))


def test_activation_check(tmp_path):
    ox.check_activation(tmp_path)  # no config: accepted
    (tmp_path / "ppo_network_config.json").write_text(json.dumps({"network_factory_kwargs": {"activation": "relu"}}))
    with pytest.raises(ox.ExportError):
        ox.check_activation(tmp_path)
    (tmp_path / "ppo_network_config.json").write_text(json.dumps({"network_factory_kwargs": {"activation": "silu"}}))
    ox.check_activation(tmp_path)


def test_onnx_graph_matches_numpy_reference(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    mean, std, layers = ox.make_synthetic_actor(seed=3)
    model = ox.build_onnx_model(mean, std, layers)
    path = ox.write_export(model, tmp_path)
    obs = np.random.default_rng(5).normal(size=(9, ox.OBS_SIZE)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.shape[1] for i in sess.get_inputs()] == [ox.OBS_SIZE]
    assert sess.get_outputs()[0].shape[1] == ox.ACTION_SIZE
    out = sess.run(None, {sess.get_inputs()[0].name: obs})[0]
    assert np.abs(out - ox.reference_forward(obs, mean, std, layers)).max() < 1e-4
    # batch independence: a single row equals the same row inside a batch
    one = sess.run(None, {sess.get_inputs()[0].name: obs[:1]})[0]
    assert np.abs(one - out[:1]).max() < 1e-5


def test_manifest_records_sha256_in_the_shape_the_native_reader_expects(tmp_path):
    pytest.importorskip("onnx")
    mean, std, layers = ox.make_synthetic_actor(seed=0)
    path = ox.write_export(ox.build_onnx_model(mean, std, layers), tmp_path, "actor.onnx")
    manifest = json.loads((tmp_path / "EXPORT_MANIFEST.json").read_text())
    entry = manifest["artifacts"]["actor.onnx"]
    assert entry["sha256"] == ox.sha256_file(path) and len(entry["sha256"]) == 64
    assert manifest["obs_size"] == 57 and manifest["action_size"] == 12
