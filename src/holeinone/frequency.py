"""Ace-count distribution for a contest: a sum of independent binomials."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class AttemptGroup:
    """A block of tee shots that share one per-shot ace probability."""

    shots: int
    p: float

    def __post_init__(self) -> None:
        if self.shots < 0:
            raise ValueError("shots must be non-negative")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must lie in [0, 1]")


def ace_count_pmf(groups: Iterable[AttemptGroup], max_aces: int = 10) -> np.ndarray:
    """P(N = k) for k = 0..max_aces; the last cell holds P(N >= max_aces)."""
    pmf = np.zeros(max_aces + 1)
    pmf[0] = 1.0
    support = np.arange(max_aces + 1)
    for group in groups:
        if group.shots == 0 or group.p == 0.0:
            continue
        g = stats.binom.pmf(support, group.shots, group.p)
        g[-1] += stats.binom.sf(max_aces, group.shots, group.p)
        full = np.convolve(pmf, g)
        pmf = full[: max_aces + 1]
        pmf[-1] += full[max_aces + 1 :].sum()
    return pmf


def prob_at_least_one(groups: Iterable[AttemptGroup]) -> float:
    """P(N >= 1), computed in log space for accuracy at tiny p."""
    groups = list(groups)
    if any(g.p == 1.0 and g.shots > 0 for g in groups):
        return 1.0
    log_none = sum(g.shots * np.log1p(-g.p) for g in groups)
    return float(-np.expm1(log_none))


def expected_aces(groups: Iterable[AttemptGroup]) -> float:
    return float(sum(g.shots * g.p for g in groups))


def mixed_ace_count_pmf(
    groups: Iterable[AttemptGroup],
    rate_cv: float,
    max_aces: int = 10,
    nodes: int = 40,
) -> np.ndarray:
    """Ace-count pmf with all probabilities scaled by a lognormal factor.

    The factor has mean 1 and CV ``rate_cv`` and models odds error shared by
    every shot. Integrated with Gauss-Hermite quadrature.
    """
    groups = list(groups)
    if rate_cv == 0.0:
        return ace_count_pmf(groups, max_aces)
    s2 = np.log1p(rate_cv**2)
    mu, s = -s2 / 2.0, np.sqrt(s2)
    x, w = np.polynomial.hermite.hermgauss(nodes)
    pmf = np.zeros(max_aces + 1)
    for xi, wi in zip(x, w, strict=True):
        m = np.exp(mu + np.sqrt(2.0) * s * xi)
        scaled = [AttemptGroup(g.shots, min(g.p * m, 1.0)) for g in groups]
        pmf += wi * ace_count_pmf(scaled, max_aces)
    return pmf / np.sqrt(np.pi)
