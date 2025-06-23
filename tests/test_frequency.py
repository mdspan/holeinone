import numpy as np
import pytest
from scipy import stats

from holeinone.frequency import (
    AttemptGroup,
    ace_count_pmf,
    expected_aces,
    mixed_ace_count_pmf,
    prob_at_least_one,
)


def test_single_group_matches_binomial():
    pmf = ace_count_pmf([AttemptGroup(144, 1 / 3000)], max_aces=5)
    expected = stats.binom.pmf(np.arange(6), 144, 1 / 3000)
    expected[-1] += stats.binom.sf(5, 144, 1 / 3000)
    np.testing.assert_allclose(pmf, expected)
    assert pmf.sum() == pytest.approx(1.0)


def test_groups_convolve_and_conserve_mass():
    groups = [AttemptGroup(100, 0.01), AttemptGroup(50, 0.05), AttemptGroup(10, 0.3)]
    pmf = ace_count_pmf(groups, max_aces=4)
    assert pmf.sum() == pytest.approx(1.0)
    assert 1 - pmf[0] == pytest.approx(prob_at_least_one(groups))


def test_prob_at_least_one_small_p_accuracy():
    groups = [AttemptGroup(1, 1e-12)]
    assert prob_at_least_one(groups) == pytest.approx(1e-12, rel=1e-9)


def test_mixture_preserves_mean_and_raises_variance():
    groups = [AttemptGroup(1000, 0.002)]
    base = ace_count_pmf(groups, max_aces=30)
    mixed = mixed_ace_count_pmf(groups, rate_cv=0.5, max_aces=30)
    k = np.arange(31)
    assert mixed @ k == pytest.approx(expected_aces(groups), rel=1e-6)
    assert mixed @ k**2 > base @ k**2
