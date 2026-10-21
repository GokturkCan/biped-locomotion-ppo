# Migration-stage validation (historical snapshot)

## 1. New repository

`biped-locomotion-ppo`, a new sibling of the private development project. Fresh git
history only; no source git database, remote, commit, push or publication is included.

## 2. Files created

See `file_inventory.txt` for every release file. The root contains README, Apache license,
third-party notices, citation, package metadata and ignore rules. Public source is under
`src/biped_locomotion/`, with configuration, tools, tests, documentation, five sanitized
result artifacts and one synthetic example. JSON defaults are packaged for installed use.

## 3. Source copied/adapted

Selected source methods from walking/unified environments are split across semantic modules;
107 method entries retain source provenance. Two module-level progress metric helpers,
command transforms/process, unified research constants, fresh actor initialization,
normalizer bypass, minimal MJX State/step/sensor helpers and the MJX cached-data auto-reset
wrapper are adapted. LICENSE is retained. Effort aggregate rows are selected/anonymized.
`extraction_manifest.json` records hashes, lines, destination methods and transformations.

## 4. Intentional exclusions

No URDF, MJCF, scene model, compiled model, mesh, source joint bounds, source initial qpos,
private name mapping, gain/scale binding defaults, checkpoint, raw trajectory, TensorBoard
log, video, supervisor PDF, model-rich fixture, full upstream tree, source git database,
Menagerie or third-party runtime binary is included. Historical classes/side branches are
represented in the lineage document rather than copied as public experiment modules.

## 5. Restricted model boundary

ModelSpec supplies external model path, ordered joint/actuator mappings, semantic roles,
contacts, sensors, reference vector/pose mapping, gains, limits, scales and metadata.
The runtime reads bounds from that external model. It never generates a replacement robot.
Source names and source spawn geometry were removed. Private files must remain outside git.

## 6. Module architecture

Model; control; observations; gait; rewards; commands; environments; training; evaluation;
analysis. Extracted kernels use explicit ancestor dispatch to retain method-routing semantics.
Public NumPy helpers and synthetic tests can run without simulation dependencies.

## 7. V24 implementation

ForwardV24 retains the active V24 method closure, 57/82 observations, asymmetric PPO,
V18 clock, V8/V9 qualifications/expected-side transitions and V24 bilateral/path terms.
The source's active V2 control override uses a settled nominal reference and fixed residual
scale. General phase-conditioned authority helpers are available but not substituted into
that active control path. Canonical historical numbers remain unchanged.

## 8. U13 implementation

UnifiedU13 reuses V24 software, adds the retained command/observation/reward shell and
64/89 contract. Training initializes actor, critic, optimizer and normalizer from scratch;
zero mean/scale kernels and inverse-softplus scale biases are retained. Assisted teachers,
routed experts and recurrent training are unsupported. The disabled teacher is an inert stub;
teacher-only diagnostics/smoothing shape are not asserted identical to historical code.

## 9. Public coverage

Control authority/clipping/PD; observation dimensions/order/privileged separation; actual
source liftoff/touchdown debounce and one-shot qualification; valid-load gates; expected-side
advance only after transfer; reward accessibility/pair bootstrap/accounting; scalar metrics;
generalized-effort units/RMS/peak; scratch/config contracts; external metadata validation;
security scanner detection; all 107 kernel imports; synthetic network initialization/inference;
MJX data auto-reset on fabricated state. No robot was built for these tests.

## 10. Checks actually run

- Package import and both YAML profile parses.
- Public pytest suite including optional CPU kernel/network/wrapper checks: **21 passed**.
- Synthetic CPU smoke, including with JAX/Flax/MuJoCo/Brax/ml-collections imports blocked.
- Static undefined-name checks.
- Pattern-based secret, personal/absolute-path, restricted-signature, binary/asset and size scans.
- Local dependency/import/network-factory sanity against the inspected installed versions.
- Static extraction comparison: **98 non-setup method bodies match after declared adaptations**;
  nine setup methods intentionally adapted; no unexpected differences.
- Content snapshot comparison: **12,220 source files/symlinks checked; zero changes**.
  Git databases, virtual environment, bytecode and tool caches were excluded from hashing;
  no commands intentionally wrote any source-project files.

At that migration stage, no network install, clean-machine dependency resolution, PPO step, optimizer update, physics
simulation, rendering, checkpoint deserialization or long evaluation was run.

## 11. Intentionally not run

**NOT RUN — PRIVATE MODEL REQUIRED:** source-versus-release physical reset/settling,
observation/control/reward/FSM trajectory parity, authorized checkpoint inference/rollout,
and exact V24/U13 physical reproduction. See reproducibility.md for individual checks.
Long training, 60-second evaluation, multi-environment/seed evaluation, GPU benchmarks and
checkpoint sweeps were excluded by the migration instruction.

## 12. Unproven semantics

Static extraction equivalence is not physical parity. External initialization, contact/sensor
frames, runtime metadata, affine-PD binding, wrapper lifecycle and full reward/observation
trajectories must be checked with authorized assets later. Historical teacher-only telemetry
changes are intentional. The U13 dual stochastic/deterministic training-evaluation logging
callback is not replicated; explicit deterministic/stochastic policies are accepted by the
bounded evaluation API. Exact V24 historical parent/seed cannot be recovered from this release.

## 13. Licensing/provenance

Upstream Apache license/copyright retained; selected dependency provenance documented.
Owner must confirm rights for uncommitted research additions, permission to publish sanitized
aggregate findings, and citation/author metadata. No license to the private robot is granted.

## 14–15. Restricted-data and path scans

Pattern scans found no restricted signatures, model/binary asset files, credential candidates,
hardcoded machine paths or files larger than 1 MB. See release_scan.json for the final inventory.
The scanner includes a synthetic detection test. Pattern scans do not prove ownership or
absence of every possible sensitive-data encoding.

## 16. Migration-stage git status

Migration snapshot: fresh main branch, no commits/remotes; files were untracked and unstaged.
No source .git directory copied. No git add, commit, remote creation or push performed.

## 17. Before first local commit

At migration time, no technical blocker was identified for a private first local commit. Files were deliberately left
unstaged for the owner's review. This historical report did not authorize committing; the subsequent final-review instruction
conditionally authorizes the first local commit. Current results are in final_release_review.md.

## 18. Before GitHub publication

Confirm research-code and aggregate-result publication rights and citation metadata. Review
the extraction/boundary/limitations; resolve dependency installation on a clean machine.
Private physical parity is still required before claiming this edition reproduces the exact
historical V24/U13 trajectories. Public contract success must not be described as that proof.
