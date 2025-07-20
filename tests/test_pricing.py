import numpy as np
import pytest

from holeinone.frequency import AttemptGroup
from holeinone.policy import Policy
from holeinone.pricing import PricingBasis, RiskLoad, price


def test_gross_premium_build_up():
    basis = PricingBasis(
        risk_load=RiskLoad.EXPECTED_VALUE,
        theta=0.2,
        rate_cv=0.0,
        fixed_expense=10,
        expense_ratio=0.3,
        profit_margin=0.1,
    )
    q = price(Policy(10_000), [AttemptGroup(100, 1e-4)], basis)
    pure = 10_000 * -np.expm1(100 * np.log1p(-1e-4))
    assert q.pure_premium == pytest.approx(pure)
    assert q.gross_premium == pytest.approx((1.2 * pure + 10) / 0.6)
    assert not q.minimum_applied


def test_minimum_premium_floor():
    basis = PricingBasis(minimum_premium=180)
    q = price(Policy(2000), [AttemptGroup(20, 1e-5)], basis)
    assert q.gross_premium == 180 and q.minimum_applied
