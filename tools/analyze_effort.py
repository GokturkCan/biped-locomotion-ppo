"""Analyze generalized-effort CSV, never actuator_force or motor sizing data."""

import argparse, csv, json
from biped_locomotion.analysis.actuator_effort import summarize_effort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="qfrc_actuator_0,... columns only")
    p.add_argument(
        "--joint-types", nargs="+", choices=["hinge", "slide"], required=True
    )
    p.add_argument("--sample-rate-hz", type=float, default=500)
    a = p.parse_args()
    with open(a.csv, newline="") as f:
        reader = csv.DictReader(f)
        names = [f"qfrc_actuator_{i}" for i in range(len(a.joint_types))]
        values = [[float(r[n]) for n in names] for r in reader]
    print(
        json.dumps(
            summarize_effort(
                values, list(range(len(names))), a.joint_types, a.sample_rate_hz
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
