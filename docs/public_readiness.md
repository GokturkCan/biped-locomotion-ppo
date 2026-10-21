# Final public-readiness checks

## Current public release

The repository is public. The owner reports supervisor authorization to publish the
project while excluding the supplied URDF and derived robot-model assets. Apache-2.0
and the retained upstream attribution apply to distributed contents, not those assets.

The per-channel effort CSV is absent from the current tree and reachable main history.
History was rewritten specifically to remove that path, producing rewritten HEAD
`02fc3a51cc84781878e378b5f853514cd0134c07`. Current main uses that clean public history;
subsequent documentation commits build on it. Other files, canonical results and commit
authorship were preserved. No trained weights, robot-model assets or media are present.

GitHub-hosted [Actions run 39654028986](https://github.com/GokturkCan/biped-locomotion-ppo/actions/runs/39654028986)
succeeded on that rewritten main commit: both **cpu** and **native** jobs passed. The
native job is the ONNX-free synthetic subset, not the full local 15-entry build. The
README's live Actions badge shows the current branch's workflow status.

Old unreferenced GitHub objects may persist outside reachable branch history under
platform retention behavior. No GitHub Support purge is claimed.

## Scope and presentation

The final pass changes documentation only. Python/native implementations, public profiles
and canonical results remain unchanged. The README leads with the RL locomotion research:
formulation, observations/actions, residual position control, gait/FSM reward engineering,
V13–V24 progression and quantitative V24 evidence. U13 remains a scratch-trained negative
command-tracking finding. Native deployment is secondary simulation/runtime integration,
not a hardware deployment claim.

Markdown uses GitHub-safe clip notation. All five Mermaid diagrams use compact LR layouts,
short labels and line breaks; the gait chain groups stages without changing their order.
Static local math/link checks passed. Actual hosted rendering remains a visual check on GitHub.

The active V24 target remains `clip(q_settled + scale * action, lower, upper)` followed by
affine-PD actuation. Actor/critic dimensions are 57/82 (V24) and 64/89 (U13); actions are
12-D, the actor emits 24 distribution parameters, PPO clips are 0.30/0.20, and U13 actor
indices 61–63 remain zero. Canonical result artifacts were unchanged by hash comparison.

## Private-validation claims

The private 60-second native-smoke row and its tick/physics-step/deadline/fall statistics
were removed. They are not public validation evidence. No private model or checkpoint was
opened and no private-model experiment was repeated.

The unsupported numerical JAX-versus-ONNX parity figure was replaced with:

> JAX and ONNX outputs agreed within the validation tolerance used during private parity testing.

The surrounding text explicitly labels this author-reported, states that no numerical
receipt is retained in the release, and distinguishes it from reproducible synthetic
ONNX Runtime-versus-NumPy tests. No numerical private result was invented or recomputed.

## Historical local validation record

| Check | Result |
|---|---|
| Current package install with test/deploy extras in an independent Python 3.12 CPU environment | passed |
| Public pytest suite, including ONNX export/runtime and native-boundary tests | 39 passed, zero skipped |
| Package / ONNX imports and dependency consistency | passed |
| Both profile config-only CLI parses | passed; no PPO started |
| Synthetic Python smoke | passed; no physics |
| CMake configure and full native compile | passed |
| Non-stepping native CTest subset, including synthetic ONNX actor generation/inference | 11 passed |
| Local Markdown targets/anchors and math macro scan | passed |
| Working-tree release scan and all reachable/reflog Git-history blobs | no findings |
| License, attribution, author/committer identity and trailers | passed technical review |

Python dependency versions for the ONNX checks: ONNX 1.23.2 / ONNX Runtime 1.31.0.
Native compilation used matching official ONNX Runtime 1.31.0 headers, MuJoCo 3.6.0,
SDL2 2.0.20, GLFW 3.3.6 and GCC 11.4. Temporary dependencies, generated random-weight
ONNX fixtures and build artifacts stayed outside the repository.

The joystick fallback initially segfaulted under the restricted sandbox. It passed under
the debugger outside that sandbox; the complete 11-test subset then passed outside it.
No joystick or algorithm code was changed to mask the environment issue.

```bash
$REVIEW_ENV/bin/python -I -B -m pytest tests -q -p no:cacheprovider
$REVIEW_ENV/bin/python -I -B tools/smoke_test.py
$REVIEW_ENV/bin/python -I -B tools/train.py --config configs/v24_forward.yaml
$REVIEW_ENV/bin/python -I -B tools/train.py --config configs/u13_scratch.yaml
$REVIEW_ENV/bin/python -I -B tools/scan_release.py
cmake -S native/v24_runtime -B "$BUILD_DIR" -DMUJOCO_ROOT="$MUJOCO_ROOT" \
  -DORT_ROOT="$ORT_ROOT" -DBIPED_PYTHON="$REVIEW_ENV/bin/python"
cmake --build "$BUILD_DIR" -j2
ctest --test-dir "$BUILD_DIR" --output-on-failure --timeout 20 \
  -E '^(test_contact_grf|synthetic_runtime_assets|smoke_runtime_synthetic|runtime_rejects_unfilled_template)$'
```

**Not run:** contact-settling/stepping test, generated-model closed-loop tests, private
physical parity, locomotion evaluation, training, rendering or hardware joystick testing.
Compiled viewer support is not a claim that on-screen rendering was tested.

## Public boundary and attribution

Review covered the working tree and every unique path/blob pair reachable from local,
remote-tracking and reflog history. The release has a single initial-assembly root;
no copied development Git history or unreachable objects were found. No robot model,
trained weights, checkpoints, raw trajectories/effort streams, runtime manifests, private
reference data, personal absolute paths, credentials or unapproved media were found.
The only native bindings template contains placeholders. Native synthetic geometry and
parameters are visibly test-local, independently authored software fixtures.
Pattern matching/manual review cannot prove ownership or detect every encoded value.

Apache-2.0 and upstream notices remain intact. External MuJoCo, ONNX Runtime, SDL2,
GLFW and OpenSSL dependencies are linked, not vendored. The undistributed private model
is outside the repository license. Author and committer identity is Göktürk Can only.

The workflow has Python test/deploy checks and a native ONNX-independent synthetic
job. Their successful hosted run is linked above. Neither trains a policy or loads the
restricted robot. The historical local validation table describes a separate dependency-rich
CPU environment and does not imply that hosted CI runs every optional test.

Private physical parity is necessary before claiming this edition exactly reproduces
the historical robot behavior; public contracts and hosted CI do not establish that claim.
Documentation snapshots in release_assembly.md, final_release_review.md and
release_validation.md retain their earlier-stage evidence and are not current publication gates.
