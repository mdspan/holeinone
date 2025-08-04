"""Monte Carlo of a season's book.

Odds are scaled by a season factor shared by all contests (rating error, which
does not diversify) and a per-contest factor (conditions on the day).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frequency import AttemptGroup
from .policy import Policy


@dataclass(frozen=True)
class Contest:
    policy: Policy
    groups: tuple[AttemptGroup, ...]
    premium: float


@dataclass(frozen=True)
class Reinsurance:
    """Per-contest excess of loss, then a quota share of the retained loss.

    Reinsurance premium is (1 + loading) * expected ceded loss.
    """

    xl_retention: float = np.inf
    xl_limit: float = np.inf
    quota_share: float = 0.0
    loading: float = 0.0

    def cede(self, loss: np.ndarray) -> np.ndarray:
        xl = np.clip(loss - self.xl_retention, 0.0, self.xl_limit)
        return xl + self.quota_share * (loss - xl)


@dataclass(frozen=True)
class SimulationResult:
    gross_loss: np.ndarray
    ceded_loss: np.ndarray
    premium: float
    reinsurance_premium: float

    @property
    def net_loss(self) -> np.ndarray:
        return self.gross_loss - self.ceded_loss

    @property
    def underwriting_result(self) -> np.ndarray:
        """Premium - reinsurance premium - net loss, before expenses."""
        return self.premium - self.reinsurance_premium - self.net_loss

    def summary(self, level: float = 0.99) -> dict[str, float]:
        r = self.underwriting_result
        net = self.net_loss
        var = float(np.quantile(net, level))
        tail = net[net >= var]
        return {
            "premium": self.premium,
            "reinsurance_premium": self.reinsurance_premium,
            "mean_gross_loss": float(self.gross_loss.mean()),
            "mean_net_loss": float(self.net_loss.mean()),
            "sd_net_loss": float(self.net_loss.std()),
            "mean_result": float(r.mean()),
            "prob_loss": float((r < 0).mean()),
            f"net_loss_var_{level:g}": var,
            f"net_loss_tvar_{level:g}": float(tail.mean()),
        }


def _lognormal_factor(rng: np.random.Generator, cv: float, size) -> np.ndarray:
    if cv == 0.0:
        return np.ones(size)
    s2 = np.log1p(cv**2)
    return rng.lognormal(-s2 / 2.0, np.sqrt(s2), size)


def simulate(
    contests: list[Contest],
    n_sims: int = 100_000,
    season_cv: float = 0.15,
    contest_cv: float = 0.20,
    reinsurance: Reinsurance = Reinsurance(),
    seed: int | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    season = _lognormal_factor(rng, season_cv, n_sims)
    gross = np.zeros(n_sims)
    ceded = np.zeros(n_sims)
    for contest in contests:
        factor = season * _lognormal_factor(rng, contest_cv, n_sims)
        aces = np.zeros(n_sims, dtype=np.int64)
        for g in contest.groups:
            aces += rng.binomial(g.shots, np.minimum(g.p * factor, 1.0))
        loss = contest.policy.losses(aces).astype(float)
        gross += loss
        ceded += reinsurance.cede(loss)
    premium = float(sum(c.premium for c in contests))
    re_premium = (1.0 + reinsurance.loading) * float(ceded.mean())
    return SimulationResult(gross, ceded, premium, re_premium)
