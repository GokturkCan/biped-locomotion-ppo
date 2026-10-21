"""Accumulate qualified event counts separately from occupancy."""

from dataclasses import asdict
from biped_locomotion.gait.events import GaitEvents


def summarize_events(events):
    totals = asdict(GaitEvents())
    for ev in events:
        for key, value in asdict(ev).items():
            totals[key] += value
    return totals
