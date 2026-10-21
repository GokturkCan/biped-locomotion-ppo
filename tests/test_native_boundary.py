"""The public native runtime must not embed robot-specific data or private artifacts."""

import json
import re
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1] / "native" / "v24_runtime"
TEXT = {".h", ".cpp", ".txt", ".json", ".md"}
FORBIDDEN_SUFFIXES = {".onnx", ".mjb", ".xml", ".urdf", ".stl", ".npz", ".npy", ".bin", ".so", ".a"}


def _files():
    return [p for p in NATIVE.rglob("*") if p.is_file() and "build" not in p.relative_to(NATIVE).parts]


def test_native_tree_has_no_model_or_weight_artifacts():
    assert NATIVE.is_dir()
    bad = [str(p.relative_to(NATIVE)) for p in _files() if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert bad == []


def test_native_sources_contain_no_robot_specific_literals():
    # Split literals so this test does not match itself.
    private_markers = ["Hip" + "_", "Knee" + "_", "Ankle" + "_", "sch" + "aft", "/home" + "/", "walking" + ".py"]
    hits = []
    for p in _files():
        if p.suffix in TEXT:
            text = p.read_text()
            hits += [f"{p.name}: {m}" for m in private_markers if m.lower() in text.lower()]
    assert hits == []


def test_runtime_layout_has_only_contract_constants():
    text = (NATIVE / "include/biped_native/runtime_layout.h").read_text()
    for name in ("kActionScale", "kKp", "kKd", "kInitQpos", "kActiveJoints", "kFloorGeom"):
        assert name not in text
    assert "kObsSize = 57" in text and "kActionSize = 12" in text


def test_bindings_template_contains_only_placeholders():
    tpl = json.loads((NATIVE / "bindings/runtime_bindings.template.json").read_text())
    strings = re.findall(r'"([^"]+)"', json.dumps(tpl))
    keys = {"joint_names", "sensors", "geoms", "root_body", "action_scale", "initial_qpos", "target_height",
            "min_root_height", "orientation", "gravity", "local_linvel", "gyro", "root_pos", "floor",
            "left_foot", "right_foot"}
    assert {s for s in strings if s not in keys} == {"REPLACE_ME"}
    assert len(tpl["joint_names"]) == 12 and tpl["target_height"] is None and tpl["min_root_height"] is None
    assert all(v is None for v in tpl["action_scale"])


def test_event_machine_has_no_compiled_root_height_target():
    assert "kTargetHeight" not in (NATIVE / "include/biped_native/event_machine.h").read_text()
