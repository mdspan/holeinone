import pytest

from holeinone import odds
from holeinone.calibration import fit_market, fit_yardage_decay
from holeinone.data import MARKET_QUOTES, PGA_TOUR_BY_YARDAGE


def test_ace_counts_match_published_odds():
    # The source's overall row disagrees with its buckets; check the buckets.
    assert sum(b.aces for b in PGA_TOUR_BY_YARDAGE) == 59
    assert sum(b.shots for b in PGA_TOUR_BY_YARDAGE) == 71_000 + 74_294


def test_pro_beta_constant_matches_fit():
    fit = fit_yardage_decay()
    assert odds.PRO_BETA == pytest.approx(fit.beta, abs=1e-4)
    assert fit.beta_se > 0


def test_market_fit_reproduces_quotes():
    fit = fit_market()
    assert fit.rmse < 0.06
    assert 1e-4 < fit.loaded_rate < 4e-4
    for q in MARKET_QUOTES:
        if q.prize >= 10_000:
            assert fit.predict(q.prize, q.golfers) == pytest.approx(q.premium, rel=0.12)


def test_implied_odds_scale_with_loading():
    fit = fit_market()
    assert fit.implied_odds(2.0) == pytest.approx(2 * fit.loaded_odds)
