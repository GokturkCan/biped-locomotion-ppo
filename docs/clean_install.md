# Clean environment validation

The final test used a new Python 3.12 virtual environment outside both repositories,
created from a standalone interpreter. The development virtual environment was not used
for the final checks. PYTHONPATH/PYTHONHOME and user-site packages were excluded, and
CPU JAX was selected. Initial unsanitized interpreter-discovery attempts were discarded;
the accepted environment had no foreign site-package paths and `pip check` passed.

## Commands

The following are the command forms executed, with local scratch and repository paths
represented by shell variables to avoid distributing personal filesystem paths.
`BASE_PYTHON` identifies the standalone Python 3.12 executable; `REVIEW_ENV`,
`REVIEW_SOURCE` and `REVIEW_WHEELS` are newly created temporary directories;
`REPO` is the release repository. The private development repository is not an input.

```bash
"$BASE_PYTHON" -m venv "$REVIEW_ENV"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I -m pip install --no-cache-dir \
  'numpy>=1.26,<3' 'PyYAML>=6,<7' 'pytest>=8,<10' build ruff
# Copy only this release into REVIEW_SOURCE, excluding git/build/cache directories.
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I -m pip install --no-cache-dir "$REVIEW_SOURCE[test,simulation]"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I -m pip wheel --no-cache-dir --no-deps \
  --wheel-dir "$REVIEW_WHEELS" "$REVIEW_SOURCE"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I -m pip install --no-deps --force-reinstall "$REVIEW_WHEEL"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  "$REVIEW_ENV/bin/python" -I -m pytest "$REPO/tests" -q -p no:cacheprovider
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I -m pip check
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I "$REPO/tools/smoke_test.py"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I "$REPO/tools/train.py" --config "$REPO/configs/v24_forward.yaml"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I "$REPO/tools/train.py" --config "$REPO/configs/u13_scratch.yaml"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$REVIEW_ENV/bin/python" -I "$REPO/tools/scan_release.py"
```

Import identity was asserted to lie under `sys.prefix`, with no development-project path
in `sys.path`. The installed package code/data was compared with the release tree.
Brax trainer import and PPO argument signatures were checked without invoking training.
No model was loaded; no optimizer/PPO update, simulation, rendering or checkpoint load ran.
Optional Warp import messages do not affect the CPU JAX backend; Warp was not installed.

Installed versions are in clean_environment_packages.json. These are an observed CPU
Python 3.12 environment snapshot, not a universal cross-platform lockfile. Exact final
outcomes and wheel digest are recorded in clean_install_result.json.
Temporary environments and wheel outputs remain outside the release repository for review.
