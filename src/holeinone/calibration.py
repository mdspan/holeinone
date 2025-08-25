"""Fits to reference data.

fit_yardage_decay: aces ~ Poisson(shots * exp(alpha - beta * (yards - 150))).
fit_market: premium = fixed + loaded_rate * prize * golfers. The loaded rate
    combines ace probability and loading; price sheets cannot separate them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .data import MARKET_QUOTES, MINIMUM_PREMIUM, PGA_TOUR_BY_YARDAGE, MarketQuote, YardageBucket
from .odds import REFERENCE_YARDS


@dataclass(frozen=True)
class YardageFit:
    odds_at_reference: float
    beta: float
    beta_se: float
    alpha_se: float

    def odds(self, yards: float) -> float:
        return self.odds_at_reference * np.exp(self.beta * (yards - REFERENCE_YARDS))


def fit_yardage_decay(
    buckets: Sequence[YardageBucket] = PGA_TOUR_BY_YARDAGE,
    min_yards: float = 120.0,
) -> YardageFit:
    used = [b for b in buckets if b.midpoint >= min_yards]
    x = np.array([b.midpoint - REFERENCE_YARDS for b in used])
    n = np.array([b.shots for b in used], dtype=float)
    k = np.array([b.aces for b in used], dtype=float)

    def nll(theta: np.ndarray) -> float:
        log_p = theta[0] - theta[1] * x
        return float(np.sum(n * np.exp(log_p) - k * log_p))

    start = np.array([np.log(k.sum() / n.sum()), 0.0])
    res = optimize.minimize(nll, start, method="BFGS")
    alpha, beta = res.x
    mu = n * np.exp(alpha - beta * x)
    info = np.array([[mu.sum(), -(mu * x).sum()], [-(mu * x).sum(), (mu * x**2).sum()]])
    cov = np.linalg.inv(info)
    return YardageFit(
        float(np.exp(-alpha)), float(beta), float(cov[1, 1] ** 0.5), float(cov[0, 0] ** 0.5)
    )


@dataclass(frozen=True)
class MarketFit:
    fixed: float
    loaded_rate: float
    rmse: float
    n_quotes: int

    @property
    def loaded_odds(self) -> float:
        return 1.0 / self.loaded_rate

    def implied_odds(self, loading_multiple: float) -> float:
        """Per-shot odds after dividing out a premium-to-expected-loss multiple."""
        return loading_multiple / self.loaded_rate

    def predict(self, prize: float, golfers: int) -> float:
        return max(self.fixed + self.loaded_rate * prize * golfers, MINIMUM_PREMIUM)


def fit_market(
    quotes: Sequence[MarketQuote] = MARKET_QUOTES,
    min_prize: float = 10_000.0,
) -> MarketFit:
    """Weighted least squares on relative error; skips quotes near the minimum."""
    used = [q for q in quotes if q.prize >= min_prize]
    exposure = np.array([q.prize * q.golfers for q in used], dtype=float)
    y = np.array([q.premium for q in used], dtype=float)
    design = np.column_stack([np.ones_like(exposure), exposure])
    w = 1.0 / y
    (fixed, rate), *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
    rel = (design @ [fixed, rate] - y) / y
    return MarketFit(float(fixed), float(rate), float(np.sqrt(np.mean(rel**2))), len(used))
