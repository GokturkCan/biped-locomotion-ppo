"""Explicit training entrypoint; no training was run during release migration."""

import argparse, json
from biped_locomotion.training.ppo_config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-spec")
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly start training using an authorized model",
    )
    a = p.parse_args()
    cfg = load_config(a.config)
    if not a.execute:
        print(
            json.dumps(
                {"mode": "configuration only; no PPO started", **cfg.train_kwargs()},
                indent=2,
            )
        )
        return
    if not a.model_spec or not a.output_dir:
        p.error("--model-spec and --output-dir required for training")
    from biped_locomotion.model.interface import ModelSpec
    from biped_locomotion.training.runner import train

    train(ModelSpec.from_json(a.model_spec), cfg, a.seed, a.output_dir)


if __name__ == "__main__":
    main()
