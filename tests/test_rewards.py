import pytest
from biped_locomotion.rewards.common import weighted_reward
from biped_locomotion.rewards.forward import bilateral_terms
from biped_locomotion.rewards.unified import command_requested
from biped_locomotion.analysis.reward_accounting import account


def test_acquisition_accessible_before_success():
    assert command_requested(0.1, 0) == 1
    total, parts = weighted_reward(
        {"deep_unload": 0.5, "support_transfer_completion": 0},
        {"deep_unload": 2, "support_transfer_completion": 10},
        0.02,
    )
    assert total == pytest.approx(0.02) and parts["support_transfer_completion"] == 0


def test_pair_bootstrap_cannot_pay_growth():
    a = bilateral_terms(0.03, 0.08, 1, 0, 1, 0, 0)
    assert a["bilateral_symmetry"] > 0 and a["imbalance_growth"] == 0
    assert bilateral_terms(0.03, 0.08, 1, 0, 1, 1, 0.2)["imbalance_growth"] == 1


def test_reward_units_and_sign():
    a = account({"event": 1, "penalty": 2}, {"event": 4, "penalty": -1}, 0.02)
    assert a["total"] == pytest.approx(0.04)
