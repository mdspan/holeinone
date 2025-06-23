"""Policy terms and the loss distribution they induce."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class PayoutBasis(StrEnum):
    """FIRST_ACE pays one prize if N >= 1; EACH_ACE pays prize * N."""

    FIRST_ACE = "first"
    EACH_ACE = "each"


@dataclass(frozen=True)
class Policy:
    prize: float
    basis: PayoutBasis = PayoutBasis.FIRST_ACE
    annuity_years: int = 1
    discount_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.prize <= 0:
            raise ValueError("prize must be positive")
        if self.annuity_years < 1:
            raise ValueError("annuity_years must be at least 1")

    @property
    def claim_cost(self) -> float:
        """Present value of one prize paid as a level annuity-due."""
        n, i = self.annuity_years, self.discount_rate
        installment = self.prize / n
        if i == 0.0 or n == 1:
            return installment * n
        v = 1.0 / (1.0 + i)
        return installment * (1.0 - v**n) / (1.0 - v)

    def losses(self, n_aces: np.ndarray | int) -> np.ndarray:
        n = np.asarray(n_aces)
        if self.basis is PayoutBasis.FIRST_ACE:
            return self.claim_cost * (n >= 1)
        return self.claim_cost * n


def loss_moments(policy: Policy, ace_pmf: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the insured loss."""
    k = np.arange(ace_pmf.size)
    x = policy.losses(k)
    mean = float(ace_pmf @ x)
    var = float(ace_pmf @ (x - mean) ** 2)
    return mean, var**0.5
