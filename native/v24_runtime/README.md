# Native C++ / ONNX runtime for the V24 policy

Publishable software portion of the native deployment work: a C++17 closed-loop runtime that
assembles the 57-value observation, runs the exported actor through ONNX Runtime, converts
actions to residual joint-position targets and steps MuJoCo, with joystick command input.
Architecture and design notes: [`docs/native_deployment.md`](../../docs/native_deployment.md).

**What is not here:** the robot model (no URDF/MJCF/`.mjb`), robot names and constants, trained
ONNX weights, checkpoints, recorded fixtures/rollouts and run manifests. Robot-specific data
enters only through a runtime **bindings JSON** you supply (below). The code was written against
one private robot; it compiles and its software contracts are tested without it.

## Layout

```
include/biped_native/   runtime_layout.h (public obs/action contract)   runtime_bindings.h (external robot binding)
                        obs_builder.h  event_machine.h  contact_utils.h  control_law.h  onnx_actor.h
                        gamepad.h  camera_controller.h  push_controller.h  telemetry.h
                        manifest_verify.h  json_lite.h  glfw_viewer.h (optional viewer)
src/                    implementations
apps/biped_v24_native.cpp   closed-loop runtime
bindings/runtime_bindings.template.json   schema with placeholders (rejected until filled in)
tests/                  C++ unit/contract tests + synthetic model builder (mjSpec, no model file)
```
Python export/test tooling: [`tools/export_onnx.py`](../../tools/export_onnx.py),
[`tools/make_synthetic_actor.py`](../../tools/make_synthetic_actor.py),
[`src/biped_locomotion/deployment/`](../../src/biped_locomotion/deployment/onnx_export.py).

## Dependencies (none vendored)

| Dependency | Needed for | Notes |
|---|---|---|
| CMake ≥ 3.16, C++17 compiler, OpenSSL (libcrypto) | everything | OpenSSL: artifact sha256 check |
| MuJoCo C library 3.x | everything | `-DMUJOCO_ROOT=<dir with include/mujoco and libmujoco.so*>`; the pip `mujoco` package directory works |
| ONNX Runtime C++ (CPU) | `onnx_actor`, runtime app, its test | `-DORT_ROOT=<unpacked release>`; if unset these targets are skipped |
| SDL2 | gamepad test, runtime app | found through pkg-config |
| GLFW3 | `--render` viewer | optional |
| Python with numpy, onnx, `biped_locomotion` importable | synthetic ONNX test fixture | `-DBIPED_PYTHON=...`; `pip install -e '.[deploy]'` |

## Build and test

```bash
pip install -e '.[test,deploy]'                       # Python side (ONNX export + tests)
MJ=$(python -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
cmake -S native/v24_runtime -B build_native -DMUJOCO_ROOT="$MJ" \
      -DORT_ROOT=/path/to/onnxruntime-linux-x64-<version> -DBIPED_PYTHON=$(which python)
cmake --build build_native -j
PYTHONPATH=src ctest --test-dir build_native --output-on-failure
```
Without `ORT_ROOT` the ONNX-independent library and its tests are built (8 tests, 9 with SDL2 for the
gamepad test); this is what CI runs. With everything available, 15 CTest entries run on synthetic data only: bindings parsing, observation
layout/formulas, control law, event machine, contact/GRF, push/camera controllers, hash
verification, ONNX wrapper vs a NumPy reference, and an end-to-end smoke run of the app.

## Runtime bindings

Copy `bindings/runtime_bindings.template.json`, fill every field from your own authorized model
and keep the file **outside git**. Fields: the 12 ordered `joint_names`; sensor names
(`orientation` quaternion wxyz, `gravity` up-vector, `local_linvel`, `gyro`, `root_pos`); `geoms`
(`floor`, `left_foot`, `right_foot`); `root_body`; 12 positive `action_scale` values;
`initial_qpos` (length = model `nq`); `target_height` and `min_root_height` in metres
(`0 < min < target`). Joint limits come from the model, gains from the model's actuators. The loader
rejects missing/placeholder/non-finite values and `ValidateModelAgainstBindings` checks the model
(sizes, scalar joints, sensor widths, named geoms). This mirrors the Python `ModelSpec`
([`docs/model_access.md`](../../docs/model_access.md)).

## Run

```bash
python tools/export_onnx.py --checkpoint <your authorized Brax checkpoint> --output-dir exports/run1
build_native/biped_v24_native --model <model.mjb|.xml> --policy exports/run1/actor.onnx \
    --bindings <your runtime_bindings.json> --manifest exports/run1/EXPORT_MANIFEST.json \
    --out out_dir --duration 60            # add --render for the optional GLFW window
```
`--manifest` makes the runtime refuse an ONNX file whose sha256 differs from the manifest.
Output: `metrics.csv` (per-tick telemetry) and `summary.json` in `--out`. Joystick: left stick,
vertical axis → forward velocity; buttons 0–4 → reset / pause / push / camera cycle / log toggle;
no controller → neutral command. The `BIPED_PUSH_AT_S` environment variable triggers one scripted push.

## Verification status

| Item | Status |
|---|---|
| Builds (MuJoCo 3.6.0, ONNX Runtime 1.27.0, SDL2, GLFW found; Ubuntu 22.04, g++) | verified locally |
| 15 synthetic-data CTest entries (full) / 9 (no ONNX Runtime, with SDL2) | verified locally; the ONNX-free subset is what CI runs (not yet observed on GitHub) |
| ONNX Runtime vs NumPy reference on a **synthetic** random-weight actor | verified (max diff < 1e-4 enforced) |
| End-to-end app smoke on synthetic assets (25 ticks) | verified; confirms the loop executes, says nothing about walking |
| GLFW viewer | compiles; on-screen rendering **not verified** in this release |
| Gamepad with physical hardware | **not verified** (only the no-controller fallback is tested) |
| Behaviour on the private robot / trained weights | not reproducible from this repository |
| Hardware deployment | **not claimed**: simulation/runtime integration only |
