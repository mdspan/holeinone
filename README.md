# holeinone

Pricing model for hole-in-one prize indemnity insurance: a tournament offers a
prize for an ace on one par-3 and insures the payout.

```sh
pip install -e '.[dev]'
pytest

holeinone quote --prize 10000 --golfers 100 --yards 165
holeinone quote --prize 1000000 --golfers 6 --annuity-years 40 --discount-rate 0.04
holeinone odds        # per-shot odds by yardage and skill
holeinone market      # model vs. published premiums
holeinone simulate --xl-retention 100000 --quota-share 0.2
```

## Model

| Module | Content |
|---|---|
| `odds.py` | `odds(y) = odds_150 * exp(beta * (y - 150))` by skill class |
| `frequency.py` | Exact ace-count pmf; lognormal mixture for parameter risk |
| `policy.py` | First-ace or every-ace payout; annuitized prizes |
| `pricing.py` | Pure premium, risk margin, expense and profit gross-up, minimum premium |
| `credibility.py` | Beta-binomial (Buhlmann) updating of the ace rate |
| `portfolio.py` | Monte Carlo of a book with excess of loss and quota share |
| `calibration.py` | Poisson fit of PGA Tour aces on yardage; fit of market premiums |

Calibration:

| Class | Odds at 150 yds | beta per yard | Source |
|---|---|---|---|
| Amateur | 12,500 | 0.0497 | Published average; Scheid's 1 in 150,000 at 200 yds |
| Pro | 2,500 | 0.0117 (s.e. 0.0056) | Poisson fit, 2014-15 PGA Tour, 59 aces |
| Low handicap, scratch | 5,000, 3,000 | interpolated | |

## Findings

Published premiums from two insurers fit `$74 + prize * golfers / 4,914`
(3.8% relative error). At 165 yards the model gives amateur odds of 1 in
26,000, so the market charges about 5.4 times expected loss under this model,
or 2.5 times under the flat 1-in-12,500 figure. Either insurers ignore most of
the yardage effect, or the amateur decay is too steep.

Default quote for $10,000, 100 golfers, 165 yards: $249 (market: $277).

## Limitations

- The PGA yardage data has few aces per bucket.
- Amateur odds by yardage rest on two published figures.
- Market quotes bundle ancillary cover into the fixed charge.
- Reinsurance terms for this line are not public; the simulation uses a loading.
