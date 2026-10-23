from pathlib import Path
import importlib.util


def test_release_gate_detects_tokens_paths_and_models(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "sample.txt").write_text(
        "gh"
        + "p_"
        + "A" * 24
        + "\n"
        + "/"
        + "home"
        + "/"
        + "example/project\n"
        + "<"
        + "robot name='test'>\n"
    )
    found = {x["kind"] for x in module.scan(tmp_path)["findings"]}
    assert {
        "credential_candidate",
        "personal_absolute_path",
        "restricted_signature",
    } <= found


def test_release_gate_detects_nested_history_and_raw_exports(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_extra", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nested = tmp_path / "downloaded" / ".git"
    nested.mkdir(parents=True)
    (nested / "config").write_text("fake nested history")
    (tmp_path / "export.csv").write_text("qpos,qvel\n1,2\n")
    found = {x["kind"] for x in module.scan(tmp_path)["findings"]}
    assert {"nested_git_history", "model_or_trajectory_columns"} <= found


def test_release_gate_waives_binary_check_only_for_approved_gif(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_gif", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    approved = "assets/demos/v24_forward_walking_preview.gif"
    small = b"GIF89a\x00\xff\x80"
    for rel in (approved, "assets/demos/other.gif", "v24_forward_walking_preview.gif", "assets/blob.dat"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(small)
    flagged = {(x["path"], x["kind"]) for x in module.scan(tmp_path)["findings"]}
    assert flagged == {
        ("assets/demos/other.gif", "binary_file"),
        ("v24_forward_walking_preview.gif", "binary_file"),
        ("assets/blob.dat", "binary_file"),
    }
    (tmp_path / approved).write_bytes(small + b"\x00" * 1_000_001)
    kinds = {x["kind"] for x in module.scan(tmp_path)["findings"] if x["path"] == approved}
    assert kinds == {"large_file"}
