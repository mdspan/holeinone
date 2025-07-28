"""Beta-binomial updating of the per-shot ace rate.

The posterior mean equals the Buhlmann estimate Z * observed + (1 - Z) * prior,
with Z = shots / (shots + a + b).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class BetaRate:
    a: float
    b: float

    @classmethod
    def from_mean(cls, mean: float, prior_shots: float) -> BetaRate:
        """Prior with the given mean, weighted as ``prior_shots`` observations."""
        if not 0.0 < mean < 1.0:
            raise ValueError("mean must lie in (0, 1)")
        return cls(mean * prior_shots, (1.0 - mean) * prior_shots)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def cv(self) -> float:
        """Usable as ``PricingBasis.rate_cv``."""
        return float(stats.beta(self.a, self.b).std() / self.mean)

    def credibility(self, shots: int) -> float:
        return shots / (shots + self.a + self.b)

    def update(self, shots: int, aces: int) -> BetaRate:
        if not 0 <= aces <= shots:
            raise ValueError("need 0 <= aces <= shots")
        return BetaRate(self.a + aces, self.b + shots - aces)

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        lo, hi = stats.beta(self.a, self.b).interval(level)
        return float(lo), float(hi)
