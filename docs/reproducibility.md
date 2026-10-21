# Reproducibility

Native C++/ONNX runtime (build and synthetic-data tests): see `native/v24_runtime/README.md`
and `docs/native_deployment.md`; the Python side needs `pip install -e '.[test,deploy]'`.

Public CPU checks:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/smoke_test.py
python tools/train.py --config configs/v24_forward.yaml
python tools/train.py --config configs/u13_scratch.yaml
```

These do not start PPO or load a robot. Optional kernel/import tests and the live backend
need `.[simulation,test]`. The pinned CPU simulation stack and public extras were installed in an independent
Python 3.12 environment during final review. Installed-wheel imports, dependency checking,
synthetic tests and smoke passed; see clean_install.md. Other Python/platform combinations
were not tested.

For authorized private use only, `tools/train.py --execute` accepts `--model-spec`,
`--output-dir`, `--config` and `--seed`. It creates new training, not the historical V24
result. No training command was executed during migration. U13 scratch initialization
is explicit; no checkpoint path is accepted for that initialization.
`tools/evaluate.py` evaluates CSV scalar streams offline. `tools/analyze_effort.py`
consumes generalized-effort CSV only. The live-policy evaluation API is documented in
`evaluation/rollout.py`; it is opt-in and has not been run with a private model.

## PRIVATE VALIDATION REQUIRED LATER

- Compare source and modular reset/reference/settled state with identical authorized metadata.
- Compare V24 57/82 and U13 64/89 observations channel by channel.
- Compare bounded actions, target positions and affine-PD parameters.
- Compare liftoff/touchdown debounce, transfer completion/timeout and expected-side transitions.
- Compare every active reward component and dt-scaled return in identical states.
- Load authorized checkpoint through the matching normalizer/network contract; compare inference.
- Validate wrapper/reset behavior and source-versus-release policy trajectories, only when authorized.

No 60-second, multi-environment or training validation is part of migration.
Historical seed/launch provenance gaps remain. Public tests do not close those gaps.

Final review: 27 tests passed with no skips in the isolated installed-wheel environment.
The accepted wheel SHA-256 and per-check commands are recorded in clean_install_result.json.
At technical-review time the first local commit was pending author identity. The owner
subsequently supplied it and authorized local release-assembly commits; see
[release assembly](release_assembly.md) for the current pass. No remote or push is part of
this release work. Publication rights remain a separate public-visibility gate.
