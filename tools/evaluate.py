"""Evaluate approved scalar streams offline; does not deserialize policies or models."""

import argparse, json, csv
from biped_locomotion.evaluation.metrics import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="time_s,world_x,body_vx,command_vx columns")
    a = p.parse_args()
    with open(a.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    stream = lambda name: [float(r[name]) for r in rows]
    print(
        json.dumps(
            summarize(
                *(stream(k) for k in ("time_s", "world_x", "body_vx", "command_vx"))
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
