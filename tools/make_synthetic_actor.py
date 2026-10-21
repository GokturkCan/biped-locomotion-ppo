"""Write a random-weight SYNTHETIC ONNX actor plus reference I/O for the native tests.

    python tools/make_synthetic_actor.py --output-dir build/synthetic

Produces ``actor.onnx``, ``EXPORT_MANIFEST.json`` and ``synthetic_io.bin`` (int32 n, n*57
float32 observations, n*12 float32 NumPy reference actions). These are test fixtures
generated from random numbers; they are not trained weights.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from biped_locomotion.deployment import onnx_export as ox


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=16)
    a = p.parse_args(argv)
    mean, std, layers = ox.make_synthetic_actor(a.seed)
    model = ox.build_onnx_model(mean, std, layers)
    ox.write_export(model, a.output_dir, extra={"synthetic": True})
    rng = np.random.default_rng(a.seed + 1)
    obs = (mean + std * rng.standard_normal((a.samples, ox.OBS_SIZE))).astype(np.float32)
    ref = ox.reference_forward(obs, mean, std, layers)
    with open(a.output_dir / "synthetic_io.bin", "wb") as f:
        f.write(np.int32(a.samples).tobytes())
        f.write(obs.astype("<f4").tobytes())
        f.write(ref.astype("<f4").tobytes())
    print(f"wrote synthetic actor and {a.samples} reference rows to {a.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
