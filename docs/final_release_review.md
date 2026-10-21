# Final technical and research release review

This is the technical-review snapshot preceding the presentation pass. The owner has
since supplied the local Git identity and authorized release-assembly commits. See
[release assembly](release_assembly.md) for current Git intent and presentation validation.

## Technical corrections

The README now leads with the research question, canonical V24 forward result, PPO/control
methodology, model exclusion and quantitative evaluation. Documentation explicitly distinguishes
active fixed-scale settled-reference control from the general phase-conditioned formulation.
The U13 public helper's critic order now matches the retained runtime, with regression tests.
U13 context/reserved channels, privileged layout, Gaussian-mean deterministic inference,
full PPO settings and bootstrap-FSM semantics are explicit. Lineage uses Problem, Hypothesis,
Modification, Physical metric, Result and Next limitation. No physical algorithm was redesigned. The 22 retained runtime modules match the
migration baseline at AST level after removing documentation strings. CSV line endings
were normalized without changing any result numbers.

Packaging uses SPDX Apache-2.0 metadata and preserved license/notice files. Pytest no longer
injects src into its import path, allowing the installed wheel to be validated independently.
Adaptation notices are retained/clarified. Release scans additionally detect nested git history,
model/trajectory CSV columns, machine identifiers, checkpoints and further binary/data formats.

## Control conclusion

Both profiles dispatch `_q_ref_phase` and `_v3_control_law` to the retained V2 overrides.
The active target is `clip(q_settled + scale * action, lower, upper)`, followed by affine PD.
Reference and scale are constant within an episode. Phase/side inputs do not modify that
control law; unload timing still participates in shaping. Phase-conditioned public helpers
are standalone methodology, not active switches in the canonical V24 execution path.

## Research consistency

- V24 observations 57/82; U13 64/89. U13 context is request/norm/start-side/transition at
  actor 57–60, with zeros at 61–63. Its critic reserved block is 86–88.
- Twelve residual position actions are parameterized by twelve Gaussian means and twelve
  scale logits. Deterministic action is tanh of the Gaussian mean, not direct torque and
  not the analytic expectation of the squashed distribution.
- PPO profiles match the requested resolved settings, including V24 clip .30 and U13 .20.
- Dense velocity reward alone did not demonstrate walking. Event-qualified acquisition,
  transfer completion, debounce, failure events and physical metrics remain distinct.
  Unknown-side bootstrap recovery is distinguished from normal opposite-side advancement.
- Configured coefficients are not asserted active; excessive_clearance is unreachable on
  the active V24 path and its stale release coefficient is zero. No fixed active-term count
  is claimed.
- Canonical V24 checkpoint and numbers are unchanged. Body velocity is not displacement/time.
- U13 reuses software with entirely fresh training initialization; it acquired gait but
  did not demonstrate successful command-conditioned locomotion.
- Research lineage is V13→V16→V17→V18→V19→V20→V21→V22→V23→V24. Earlier implementation ancestry
  and side branches are separate. V18 is not presented as strict causal proof.
- Effort is qfrc_actuator at 500 Hz, peak/p99/RMS, with rotational N*m and prismatic N.
  Public aggregates contain no raw trajectories, private mappings or force limits.

## Licensing and attribution

The retained code's Apache-2.0 notices and full license are preserved; adapted files identify
modification. No incompatible vendored code was found. Newly authored release boundaries,
utilities, synthetic tests and docs are distinguished from adapted research/upstream helpers.
Brax and the simulation stack are external dependencies, not copied code bundles.
Apache-2.0 section 4 supports derivative distribution with its license/notice conditions:
https://www.apache.org/licenses/LICENSE-2.0
Research-code ownership and institutional publication authority are not established by an
upstream header. The owner was asked to confirm authority for the original code and aggregate
results; no private-model license is implied. See THIRD_PARTY_NOTICES.md.

## Validation

Independent installed-wheel Python 3.12 environment; CPU only. Package/dependency installation,
imports, both configuration parses, **27 tests passed with zero skips**, smoke, static undefined-name checks
and release scan passed. All package code/data matched the installed distribution; tests
did not import the release src through pytest path injection. clean_install.md contains command
forms and clean_install_result.json records outcomes. Additional observation tests use fabricated
vectors and sensor readers, never a robot description or simulator model.

**NOT RUN — PRIVATE MODEL REQUIRED:** physical reset/settling, control/reward/FSM trajectory parity,
authorized checkpoint inference, and exact V24/U13 reproduction. No training, physics simulation,
long evaluation, rendering or sweep was run. The development archive and private assets were not
inputs to this review.

## Publication and commit gates at technical-review time

Pattern scans cover restricted signatures/assets, machine paths/identifiers, token patterns,
raw model-rich exports, binary/large artifacts and nested git history. They cannot establish
ownership or rule out every encoded sensitive value. The small aggregate records were reviewed
for schema and canonical-number consistency.

The first local commit is conditional on passing technical checks and a valid author identity.
No Git author name/email was configured at review time; the owner was asked to provide one.
Nothing will be staged or committed while a required identity or critical ownership decision
remains unresolved. No remote, push or publication is authorized by this review.
Before public visibility, confirm rights/attribution and accept the documented reproduction
limits; physical parity remains necessary before claiming exact release-edition reproduction.
The final assistant report supplies the actual git status and commit hash, if any.

## Historical technical-review disposition

Technical gates passed: isolated wheel installation, dependency consistency, 27 public tests
with no skips, smoke, both config-only CLIs, runtime imports, installed-code/data identity,
publication scan and whitespace review. Canonical result files remain unchanged in value.

Git remains a fresh main branch with no objects/commits/remotes. Files are untracked and
unstaged. `git var GIT_AUTHOR_IDENT` confirmed no usable author identity. Name/email were
requested rather than fabricated. Release-authority confirmation was also requested because
an upstream license header does not establish ownership of original research additions.
No first commit was attempted while these inputs were pending.

Technical eligibility for a private release repository is established; no repository was
created remotely. Before public visibility, confirm original-code/aggregate publication
rights and citation metadata. Private physical validation remains required only for a claim
that the modular edition reproduces the exact historical robot trajectories.
