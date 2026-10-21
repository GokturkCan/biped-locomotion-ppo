# Research presentation and release assembly

Historical initial-assembly snapshot: commit/push statements and permission checks below
describe that earlier stage, not current release blockers. The repository is now public;
the owner reports supervisor authorization with supplied model assets excluded. See
[current public-release status](public_readiness.md).

This pass changes presentation, navigation and author metadata only. The validated
algorithms, public configurations and sanitized canonical result values are unchanged.
The README contains a centered scope/result header, factual dependency badges, three
Mermaid diagrams, semantic observation groups and linked research detail. At that stage, the tests badge
labeled public contracts; the current README instead uses a live GitHub Actions badge.
No robot media, model assets, new physical measurements or fabricated history are added.

## Quick validation

Validation uses the independent installed-wheel CPU environment from technical review,
outside both repositories, rather than the development environment. Commands use the
installed package, isolated Python mode and disabled bytecode writes. Environment-local
paths below are placeholders, not repository defaults.

```bash
$REVIEW_ENV/bin/python -I -B -c 'import biped_locomotion'
$REVIEW_ENV/bin/python -I -B -m pip check
$REVIEW_ENV/bin/python -I -B -m pytest $REPO/tests -q -p no:cacheprovider
$REVIEW_ENV/bin/python -I -B $REPO/tools/smoke_test.py
$REVIEW_ENV/bin/python -I -B $REPO/tools/train.py --config $REPO/configs/v24_forward.yaml
$REVIEW_ENV/bin/python -I -B $REPO/tools/train.py --config $REPO/configs/u13_scratch.yaml
$REVIEW_ENV/bin/python -I -B $REPO/tools/scan_release.py
```

The presentation-pass results are recorded in [presentation_validation.json](presentation_validation.json).
Local Markdown targets are checked, and hashes verify unchanged source, configurations,
results, tests and tools. Pattern scanning covers model signatures/extensions, personal
paths, credentials, machine identifiers, large/binary files and generated artifacts.
It cannot prove provenance, ownership or absence of every encoded sensitive value.
No physical simulation, training, evaluation or rendering is part of this pass.

## Git assembly

The first release is assembled on one `main` branch with logical metadata, implementation,
test and documentation commits. These are release-assembly commits, not reconstructions
of V13–V24 history. Only repository-local identity is configured: Göktürk Can,
132710028+GokturkCan@users.noreply.github.com. No AI author or trailer is added.
No remote is created and nothing is pushed. The final delivery report records hashes and
actual status; historical migration/technical-review records describe their earlier state.

## Historical publication review

Technical checks validate public contracts, not the exact historical V24 trajectory.
Authorized private-model parity is required before claiming exact physical reproduction
of this edition. Publication authorization was pending at assembly time; the owner now
reports supervisor authorization for publication with supplied model assets excluded.
The private robot remains outside Apache-2.0 repository contents and no license to it
is implied. See [third-party notices](../THIRD_PARTY_NOTICES.md).
