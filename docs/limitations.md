# Limitations

- Private robot/model/checkpoint data are absent; public checks cannot reproduce V24 physically.
- Extracted source equations and call ordering have not been proven equivalent on the private model.
- External boundary replaces source model loading, mapping, initial qpos and actuation binding.
  Incorrect private metadata changes physics even if dimensions match.
- U13 historical teacher assistance is unsupported. Its disabled teacher is a zero stub; diagnostic
  teacher-only telemetry and smoothing-state shape are not claimed to be identical.
- Unreachable historical setup assignments were removed; internal source-method IDs remain to
  preserve dispatch provenance rather than recreate historical public experiment classes.
- Some shared code still contains inert later-stage branches. Only the supplied V24/U13 profiles
  are supported; arbitrary historical stage overrides are outside the tested contract.
- V18 evidence is single-seed with simultaneous timing changes. U13 architecture/reward comparisons
  do not isolate every causal factor. V24 original training seed and restore path are unpersisted.
- Lateral/yaw/backward/omnidirectional success, hardware transfer, rough terrain, push robustness
  and seed robustness are not established by this artifact.
- Exact generalized effort is simulation-dependent and is not a motor specification.

- Final clean-environment validation covers the installed public package and synthetic kernels
  on CPU, not model physics. Dependency installation alone does not establish physical parity.

- Native deployment: the C++ runtime and ONNX export are verified only on synthetic data here. The trained
  ONNX weights, robot model and bindings are private, so closed-loop behaviour on the real robot model is
  not reproducible from this repository. The GLFW viewer compiles but on-screen rendering is unverified,
  the gamepad was tested only without a physical controller, and no hardware deployment is claimed.
