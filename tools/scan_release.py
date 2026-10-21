"""Static release gate: no credentials, private signatures, local paths or large assets."""

from pathlib import Path
import re, json, sys


def scan(root):
    root = Path(root)
    findings = []
    files = 0
    total = 0
    largest = []
    # Split literals keep the scanner from matching its own private-signature rules.
    signatures = [
        "Hip" + "_EI" + "_L",
        "Knee" + "_TJ" + "_L",
        "Foot" + "_link" + "_L",
        "<" + "robot",
        "<" + "mujoco",
        "full" + "inertia",
        "0.103" + "529",
    ]
    forbidden = {
        ".urdf",
        ".xml",
        ".mjb",
        ".stl",
        ".obj",
        ".dae",
        ".npz",
        ".onnx",
        ".mp4",
        ".zip",
        ".pdf",
        ".npy",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".ckpt",
        ".ocdbt",
        ".bin",
        ".h5",
        ".hdf5",
        ".tar",
        ".gz",
    }
    local = re.compile(
        "|".join(
            re.escape(p)
            for p in (
                "/" + "home" + "/",
                "/" + "Users" + "/",
                "C:" + chr(92) + "Users" + chr(92),
                "/" + "tmp" + "/",
                "/" + "mnt" + "/",
                "/" + "media" + "/",
                "/" + "opt" + "/",
                "/" + "usr" + "/",
            )
        )
    )
    token = re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,})|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    assignment = re.compile(
        r"""(?i)(?:api[_-]?key|auth[_-]?token|access[_-]?token|secret[_-]?key|password)\s*["']?\s*[:=]\s*["']([^"'\n]{8,})["']"""
    )
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if ".git" in rel.parts and rel.parts[0] != ".git":
            findings.append({"path": str(rel), "kind": "nested_git_history"})
            continue
        if any(
            part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
            for part in rel.parts
        ):
            continue
        if p.is_symlink():
            findings.append({"path": str(rel), "kind": "symlink"})
            continue
        if not p.is_file():
            continue
        if any(
            part in {"checkpoints", "logs", "wandb"} for part in rel.parts
        ) or p.name.startswith("events.out.tfevents"):
            findings.append({"path": str(rel), "kind": "generated_private_artifact"})
        b = p.read_bytes()
        files += 1
        total += len(b)
        largest.append((len(b), str(rel)))
        if p.suffix.lower() in forbidden:
            findings.append({"path": str(rel), "kind": "forbidden_asset_extension"})
        if len(b) > 1_000_000:
            findings.append({"path": str(rel), "kind": "large_file"})
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"path": str(rel), "kind": "binary_file"})
            continue
        if p.suffix.lower() == ".csv":
            import csv

            header = next(csv.reader(s.splitlines()), [])
            if set(header) & {
                "qpos",
                "qvel",
                "joint_target",
                "joint_range_lo",
                "joint_range_hi",
                "axis_x",
                "axis_y",
                "axis_z",
                "actuator_force_limit",
            }:
                findings.append(
                    {"path": str(rel), "kind": "model_or_trajectory_columns"}
                )
        for i, line in enumerate(s.splitlines(), 1):
            if any(x.lower() in line.lower() for x in signatures):
                findings.append(
                    {"path": str(rel), "line": i, "kind": "restricted_signature"}
                )
            machine_ids = ("gokturk" + "can", "230" + "611501")
            if any(v in line for v in machine_ids):
                findings.append(
                    {"path": str(rel), "line": i, "kind": "machine_identifier"}
                )
            if local.search(line):
                findings.append(
                    {"path": str(rel), "line": i, "kind": "personal_absolute_path"}
                )
            if token.search(line) or assignment.search(line):
                findings.append(
                    {"path": str(rel), "line": i, "kind": "credential_candidate"}
                )
    return {
        "files_scanned": files,
        "total_bytes": total,
        "largest_files": [
            {"bytes": n, "path": p} for n, p in sorted(largest, reverse=True)[:5]
        ],
        "findings": findings,
        "passed": not findings,
        "limits": "Pattern-based text/file scan; not a proof of ownership or absence of all possible sensitive data.",
    }


if __name__ == "__main__":
    result = scan(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
