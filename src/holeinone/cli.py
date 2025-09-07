"""Command-line interface: quote, odds, market, simulate."""

from __future__ import annotations

import argparse

import numpy as np

from .calibration import fit_market, fit_yardage_decay
from .odds import SKILLS, field_groups
from .policy import PayoutBasis, Policy
from .portfolio import Contest, Reinsurance, simulate
from .pricing import PricingBasis, RiskLoad, price


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _add_basis_args(p: argparse.ArgumentParser) -> None:
    d = PricingBasis()
    g = p.add_argument_group("pricing basis")
    g.add_argument("--risk-load", choices=[r.value for r in RiskLoad], default=d.risk_load.value)
    g.add_argument("--theta", type=float, default=d.theta, help="risk-load parameter")
    g.add_argument(
        "--rate-cv",
        type=float,
        default=d.rate_cv,
        help="coefficient of variation of the odds (parameter risk)",
    )
    g.add_argument("--fixed-expense", type=float, default=d.fixed_expense)
    g.add_argument("--expense-ratio", type=float, default=d.expense_ratio)
    g.add_argument("--profit-margin", type=float, default=d.profit_margin)
    g.add_argument("--minimum-premium", type=float, default=d.minimum_premium)


def _basis(args: argparse.Namespace) -> PricingBasis:
    return PricingBasis(
        risk_load=RiskLoad(args.risk_load),
        theta=args.theta,
        rate_cv=args.rate_cv,
        fixed_expense=args.fixed_expense,
        expense_ratio=args.expense_ratio,
        profit_margin=args.profit_margin,
        minimum_premium=args.minimum_premium,
    )


def cmd_quote(args: argparse.Namespace) -> None:
    field = {
        "amateur": args.golfers,
        "low_handicap": args.low_handicap,
        "scratch": args.scratch,
        "pro": args.pros,
    }
    groups = field_groups(field, args.yards, args.shots)
    policy = Policy(args.prize, PayoutBasis(args.basis), args.annuity_years, args.discount_rate)
    q = price(policy, groups, _basis(args))

    print(f"Hole: {args.yards:g} yards, {args.shots} shot(s) per golfer")
    for name, count in field.items():
        if count:
            odds = 1 / SKILLS[name].probability(args.yards)
            print(f"  {count:>4} {name:<13} 1 in {odds:,.0f} per shot")
    print(f"Prize: {_money(args.prize)} ({policy.basis.value} ace)")
    print(f"Claim cost: {_money(policy.claim_cost)}")
    print()
    print(f"P(at least one ace)   {q.prob_payout:.4%}  (1 in {1 / q.prob_payout:,.0f})")
    print(f"Expected aces         {q.expected_aces:.5f}")
    print(f"Loss std. deviation   {_money(q.loss_sd)}")
    print()
    for k, v in q.components.items():
        print(f"  {k.replace('_', ' '):<20} {_money(v):>12}")
    note = "  (minimum premium applied)" if q.minimum_applied else ""
    print(f"Gross premium          {_money(q.gross_premium):>12}{note}")
    print(f"Expected loss ratio    {q.loss_ratio:>12.1%}")
    if set(k for k, v in field.items() if v) == {"amateur"} and args.shots == 1:
        m = fit_market()
        print(
            f"Market benchmark       {_money(m.predict(args.prize, args.golfers)):>12}"
            "  (fit to published price sheets, holes of about 165+ yards)"
        )


def cmd_odds(args: argparse.Namespace) -> None:
    yards = np.arange(args.start, args.stop + 1, args.step)
    print(f"{'yards':>6}" + "".join(f"{n:>14}" for n in SKILLS))
    for y in yards:
        print(f"{y:>6}" + "".join(f"{1 / s.probability(y):>14,.0f}" for s in SKILLS.values()))
    print("\nValues are 1-in-N odds per tee shot.")


def cmd_market(args: argparse.Namespace) -> None:
    m = fit_market()
    y = fit_yardage_decay()
    print("Published premiums ~ fixed + loaded_rate * prize * golfers")
    print(f"  fixed charge         {_money(m.fixed)}")
    print(f"  loaded rate          1 in {m.loaded_odds:,.0f} per golfer per prize dollar")
    print(f"  relative RMSE        {m.rmse:.1%} over {m.n_quotes} quotes")
    print()
    print("Per-shot odds implied by the market after removing a loading multiple:")
    for mult in (1.5, 2.0, 2.5, 3.0):
        print(f"  {mult:.1f}x  ->  1 in {m.implied_odds(mult):,.0f}")
    model = 1 / SKILLS["amateur"].probability(args.yards)
    print(f"\nModel amateur odds at {args.yards:g} yards: 1 in {model:,.0f}")
    print(f"Loading multiple that reconciles the two: {model / m.loaded_odds:.2f}x")
    print()
    print("PGA Tour yardage fit (Poisson, 2014-15, buckets of 120+ yards):")
    print(f"  odds at 150 yards    1 in {y.odds_at_reference:,.0f}")
    print(
        f"  decay per yard       {y.beta:.4f} (s.e. {y.beta_se:.4f}),"
        f" odds x{np.exp(50 * y.beta):.2f} per 50 yards"
    )


def _sample_book(n: int, basis: PricingBasis, rng: np.random.Generator) -> list[Contest]:
    sizes = rng.choice([72, 100, 144], size=n, p=[0.3, 0.4, 0.3])
    yards = rng.integers(150, 191, size=n)
    prizes = rng.choice(
        [10_000, 25_000, 50_000, 100_000, 1_000_000], size=n, p=[0.45, 0.25, 0.18, 0.10, 0.02]
    )
    contests = []
    for golfers, y, prize in zip(sizes, yards, prizes, strict=True):
        policy = (
            Policy(float(prize), annuity_years=40, discount_rate=0.04)
            if prize >= 1_000_000
            else Policy(float(prize))
        )
        groups = tuple(field_groups({"amateur": int(golfers)}, float(y)))
        q = price(policy, list(groups), basis)
        contests.append(Contest(policy, groups, q.gross_premium))
    return contests


def cmd_simulate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    basis = _basis(args)
    book = _sample_book(args.contests, basis, rng)
    re = Reinsurance(args.xl_retention, args.xl_limit, args.quota_share, args.re_loading)
    res = simulate(book, args.sims, args.season_cv, args.contest_cv, re, seed=args.seed)
    gross_expenses = sum(c.premium for c in book) * basis.expense_ratio + basis.fixed_expense * len(
        book
    )
    print(f"Book: {len(book)} contests, {args.sims:,} simulated seasons")
    for k, v in res.summary(args.level).items():
        shown = f"{v:.2%}" if k == "prob_loss" else _money(v)
        print(f"  {k:<24} {shown:>16}")
    print(f"  {'expenses (not in result)':<22} {_money(gross_expenses):>16}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="holeinone", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("quote", help="price one contest")
    q.add_argument("--prize", type=float, required=True)
    q.add_argument("--golfers", type=int, default=0, help="number of amateur golfers")
    q.add_argument("--low-handicap", type=int, default=0)
    q.add_argument("--scratch", type=int, default=0)
    q.add_argument("--pros", type=int, default=0)
    q.add_argument("--yards", type=float, default=165.0)
    q.add_argument("--shots", type=int, default=1, help="tee shots per golfer")
    q.add_argument("--basis", choices=[b.value for b in PayoutBasis], default="first")
    q.add_argument("--annuity-years", type=int, default=1)
    q.add_argument("--discount-rate", type=float, default=0.0)
    _add_basis_args(q)
    q.set_defaults(func=cmd_quote)

    o = sub.add_parser("odds", help="odds table by yardage and skill")
    o.add_argument("--start", type=int, default=120)
    o.add_argument("--stop", type=int, default=220)
    o.add_argument("--step", type=int, default=10)
    o.set_defaults(func=cmd_odds)

    m = sub.add_parser("market", help="compare the model with published premiums")
    m.add_argument("--yards", type=float, default=165.0)
    m.set_defaults(func=cmd_market)

    s = sub.add_parser("simulate", help="Monte Carlo of a synthetic book")
    s.add_argument("--contests", type=int, default=500)
    s.add_argument("--sims", type=int, default=20_000)
    s.add_argument("--season-cv", type=float, default=0.15)
    s.add_argument("--contest-cv", type=float, default=0.20)
    s.add_argument("--xl-retention", type=float, default=np.inf)
    s.add_argument("--xl-limit", type=float, default=np.inf)
    s.add_argument("--quota-share", type=float, default=0.0)
    s.add_argument("--re-loading", type=float, default=0.3)
    s.add_argument("--level", type=float, default=0.99)
    s.add_argument("--seed", type=int, default=2026)
    _add_basis_args(s)
    s.set_defaults(func=cmd_simulate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
