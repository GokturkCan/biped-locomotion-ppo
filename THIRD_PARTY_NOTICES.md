# Provenance and licensing

The included Apache-2.0 LICENSE is retained from MuJoCo Playground.
Adapted source retains `Copyright 2025 DeepMind Technologies Limited` notices.
Upstream: https://github.com/google-deepmind/mujoco_playground
Research working-tree base revision: `462053ac3db62156a708dceb29be9375f16b07e8`.
The research additions were uncommitted in that working tree; the upstream revision
alone does not identify those additions. `docs/extraction_manifest.json` records source
file hashes, source methods, line references and extraction adaptations.

Adapted research code: selected reachable methods from `walking.py` and
`unified_v1.py`; command transforms/process; unified constants; actor preprocessing
and initialization. Selected `State`, stepping and sensor helpers derive from upstream
`mjx_env.py`. The selected MJX auto-reset wrapper derives from upstream `wrapper.py`. The modular boundary, public utilities, tests and documentation were
created for this release edition. No robot-description license is granted here.

NumPy, PyYAML, JAX, Flax, MuJoCo/MJX, Brax and ml-collections are external
requirements, not vendored source or binaries. Preserve their own notices when
redistributing dependency bundles. No Menagerie assets or ONNX Runtime binaries are included. The native runtime under
`native/v24_runtime/` links, but does not vendor, the MuJoCo C library, ONNX Runtime, SDL2, GLFW (optional)
and OpenSSL; their licenses apply to any binaries you build or distribute.

Before publication, the project owner must confirm ownership/permission for the
research additions and approved aggregate findings. Author/citation metadata identify Göktürk Can.
Existing source headers do not establish ownership of every research addition.

## Final composition review

Apache-2.0 is consistent with the retained Apache-licensed upstream helpers and the
intended license of the original research additions, subject to the owner's authority.
No incompatible vendored code was identified. Brax is used as an external dependency;
selected actor-head/preprocessing edits are research code operating through its API.
The model boundary, public assembly/analysis utilities, synthetic fixtures/tests and
release documentation are newly authored release work. Retained upstream headers are
preserved; adapted files identify their modifications. See Apache-2.0 section 4:
https://www.apache.org/licenses/LICENSE-2.0

This review establishes the included notices and technical composition, not ownership
of research work subject to an institutional or supervisor agreement. The restricted
model is outside this license and is not part of the release.
