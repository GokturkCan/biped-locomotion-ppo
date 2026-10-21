"""Run after installing this repository, or with PYTHONPATH=src."""

import json
from biped_locomotion.smoke import run

print(json.dumps(run(), indent=2))
