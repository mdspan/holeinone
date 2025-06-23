import numpy as np
import pytest

from holeinone.policy import PayoutBasis, Policy, loss_moments


def test_annuity_due_present_value():
    p = Policy(prize=1_000_000, annuity_years=20, discount_rate=0.05)
    v = 1 / 1.05
    assert p.claim_cost == pytest.approx(50_000 * (1 - v**20) / (1 - v))
    assert Policy(prize=1000).claim_cost == 1000


def test_payout_basis():
    pmf = np.array([0.5, 0.3, 0.2])
    first, _ = loss_moments(Policy(100, PayoutBasis.FIRST_ACE), pmf)
    each, _ = loss_moments(Policy(100, PayoutBasis.EACH_ACE), pmf)
    assert first == pytest.approx(50)
    assert each == pytest.approx(70)
