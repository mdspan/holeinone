import pytest

from holeinone.credibility import BetaRate


def test_credibility_matches_posterior_mean():
    prior = BetaRate.from_mean(1 / 3000, prior_shots=30_000)
    post = prior.update(shots=20_000, aces=12)
    z = prior.credibility(20_000)
    assert post.mean == pytest.approx(z * 12 / 20_000 + (1 - z) * prior.mean)
