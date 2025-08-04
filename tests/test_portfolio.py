import numpy as np
import pytest

from holeinone.frequency import AttemptGroup
from holeinone.policy import Policy
from holeinone.portfolio import Contest, Reinsurance, simulate


def test_simulation_mean_and_reinsurance():
    c = Contest(Policy(10_000), (AttemptGroup(100, 1e-3),), premium=300)
    re = Reinsurance(quota_share=0.5, loading=0.1)
    res = simulate([c] * 20, n_sims=50_000, season_cv=0, contest_cv=0, reinsurance=re, seed=1)
    expected = 20 * 10_000 * -np.expm1(100 * np.log1p(-1e-3))
    assert res.gross_loss.mean() == pytest.approx(expected, rel=0.02)
    np.testing.assert_allclose(res.ceded_loss, 0.5 * res.gross_loss)
    assert res.reinsurance_premium == pytest.approx(1.1 * res.ceded_loss.mean())


def test_excess_of_loss_caps_net_per_contest():
    re = Reinsurance(xl_retention=25_000, xl_limit=75_000)
    ceded = re.cede(np.array([0.0, 20_000, 50_000, 200_000]))
    np.testing.assert_allclose(ceded, [0, 0, 25_000, 75_000])
