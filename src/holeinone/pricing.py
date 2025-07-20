"""Premium principles and the build-up from pure premium to gross premium."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .data import MINIMUM_PREMIUM
from .frequency import AttemptGroup, mixed_ace_count_pmf
from .policy import Policy, loss_moments


class RiskLoad(StrEnum):
    """Risk margin: theta times E[L], SD[L], or Var[L]."""

    EXPECTED_VALUE = "expected_value"
    STD_DEV = "std_dev"
    VARIANCE = "variance"


@dataclass(frozen=True)
class PricingBasis:
    """G = pure + margin + fixed_expense + (expense_ratio + profit_margin) * G,
    floored at ``minimum_premium``."""

    risk_load: RiskLoad = RiskLoad.STD_DEV
    theta: float = 0.10
    rate_cv: float = 0.25
    fixed_expense: float = 50.0
    expense_ratio: float = 0.30
    profit_margin: float = 0.10
    minimum_premium: float = MINIMUM_PREMIUM

    def __post_init__(self) -> None:
        if self.expense_ratio + self.profit_margin >= 1.0:
            raise ValueError("expense_ratio + profit_margin must be below 1")


@dataclass(frozen=True)
class Quote:
    prob_payout: float
    expected_aces: float
    pure_premium: float
    loss_sd: float
    risk_margin: float
    gross_premium: float
    minimum_applied: bool
    components: dict[str, float] = field(default_factory=dict)

    @property
    def loss_ratio(self) -> float:
        return self.pure_premium / self.gross_premium


def risk_margin(principle: RiskLoad, theta: float, mean: float, sd: float) -> float:
    if principle is RiskLoad.EXPECTED_VALUE:
        return theta * mean
    if principle is RiskLoad.STD_DEV:
        return theta * sd
    return theta * sd**2


def price(
    policy: Policy,
    groups: list[AttemptGroup],
    basis: PricingBasis = PricingBasis(),
    max_aces: int = 10,
) -> Quote:
    pmf = mixed_ace_count_pmf(groups, basis.rate_cv, max_aces)
    mean, sd = loss_moments(policy, pmf)
    margin = risk_margin(basis.risk_load, basis.theta, mean, sd)
    variable = basis.expense_ratio + basis.profit_margin
    technical = (mean + margin + basis.fixed_expense) / (1.0 - variable)
    gross = max(technical, basis.minimum_premium)
    return Quote(
        prob_payout=float(1.0 - pmf[0]),
        expected_aces=float(pmf @ range(pmf.size)),
        pure_premium=mean,
        loss_sd=sd,
        risk_margin=margin,
        gross_premium=gross,
        minimum_applied=gross > technical,
        components={
            "pure_premium": mean,
            "risk_margin": margin,
            "fixed_expense": basis.fixed_expense,
            "variable_expense": basis.expense_ratio * technical,
            "profit": basis.profit_margin * technical,
        },
    )
