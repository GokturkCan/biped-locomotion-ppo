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


def test_release_gate_waives_binary_and_size_only_for_approved_gif(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_gif", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    approved = "assets/demos/v24_forward_walking_preview.gif"
    others = (
        "assets/demos/other.gif",
        "v24_forward_walking_preview.gif",
        "assets/blob.dat",
    )

    def scan_with(size):
        payload = b"GIF89a\x00\xff\x80" + b"\x00" * size
        for rel in (approved, *others, "assets/model.urdf"):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(payload)
        found = {}
        for x in module.scan(tmp_path)["findings"]:
            found.setdefault(x["path"], set()).add(x["kind"])
        return found

    small = scan_with(10)
    assert approved not in small
    for rel in others:
        assert small[rel] == {"binary_file"}
    assert small["assets/model.urdf"] >= {"forbidden_asset_extension"}

    big = scan_with(1_000_001)
    assert approved not in big
    for rel in others:
        assert big[rel] == {"binary_file", "large_file"}
    assert big["assets/model.urdf"] >= {"forbidden_asset_extension", "large_file"}
