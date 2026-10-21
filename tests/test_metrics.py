import pytest
from biped_locomotion.evaluation.metrics import summarize
from biped_locomotion.analysis.actuator_effort import summarize_effort


def test_body_velocity_distinct_from_world_rate():
    m = summarize(
        [0, 1, 3], [0, 1, 3], [0.2, 0.4, 0.9], [0.3, 0.3, 0.3], single_support=[1, 0, 0]
    )
    assert m["world_displacement_rate_m_s"] == 1
    assert m["mean_body_vx_m_s"] == pytest.approx(1 / 3)
    assert m["single_support_fraction"] == pytest.approx(1 / 3)


def test_effort_uses_generalized_dofs_and_units():
    rows = summarize_effort([[0, 3, -4], [0, -3, 4]], [1, 2], ["hinge", "slide"])
    assert rows[0]["unit"] == "N*m" and rows[1]["unit"] == "N"
    assert rows[0]["rms"] == 3 and rows[1]["peak_absolute"] == 4
    assert rows[0]["sample_rate_hz"] == 500


def test_empty_metrics_rejected():
    with pytest.raises(ValueError):
        summarize([], [], [], [])
