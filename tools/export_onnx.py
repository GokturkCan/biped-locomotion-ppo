"""Export an authorized Brax PPO checkpoint's deterministic actor to ONNX.

    python tools/export_onnx.py --checkpoint /path/to/checkpoint_dir --output-dir exports/run1

You must supply the checkpoint; none is distributed with this repository. Requires the
``simulation`` extra (brax/jax) and ``onnx``. The output directory receives ``actor.onnx``
and ``EXPORT_MANIFEST.json`` (sha256) for the native runtime's hash check. The tool does
not verify the exported graph against a robot model; see docs/native_deployment.md.
"""

import argparse
import sys
from pathlib import Path

from biped_locomotion.deployment import onnx_export as ox


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="Brax PPO checkpoint directory (user supplied)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--filename", default="actor.onnx")
    p.add_argument("--parity-samples", type=int, default=64, help="random inputs for the NumPy-vs-ONNX Runtime check (0 disables)")
    a = p.parse_args(argv)
    if not a.checkpoint.is_dir():
        print(f"checkpoint directory not found: {a.checkpoint}\nSupply an authorized checkpoint; none is distributed.", file=sys.stderr)
        return 2
    mean, std, layers = ox.load_brax_checkpoint(a.checkpoint)
    model = ox.build_onnx_model(mean, std, layers)
    path = ox.write_export(model, a.output_dir, a.filename)
    print(f"wrote {path}")
    if a.parity_samples:
        import numpy as np
        import onnxruntime as ort

        rng = np.random.default_rng(0)
        obs = (mean + std * rng.standard_normal((a.parity_samples, ox.OBS_SIZE))).astype(np.float32)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: obs})[0]
        diff = float(np.abs(out - ox.reference_forward(obs, mean, std, layers)).max())
        print(f"max |ONNX Runtime - NumPy reference| over {a.parity_samples} random inputs: {diff:.3e}")
        if diff > 1e-4:
            print("parity check FAILED", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
