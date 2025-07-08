"""Calibration data (retrieved October 2026).

PGA_TOUR_BY_YARDAGE: 2014-15 seasons, aces = shots / published odds.
    https://probablegolfinstruction.com/golf-science-newsletter-hole-in-one-odds-pga-yardage.html
MARKET_QUOTES: amateur fields, one prize hole.
    HIOI  https://www.holeinoneinternational.com/how-much-does-hole-in-one-insurance-cost/
          (holes of 165+ yards)
    USHIO https://www.holeinoneinsurance.com/hole-in-one-insurance-prizes.html
"""

from __future__ import annotations

from typing import NamedTuple


class YardageBucket(NamedTuple):
    label: str
    midpoint: float
    shots: int
    aces: int


# The "<120" bucket is excluded from the fit (tour short holes use tucked pins).
PGA_TOUR_BY_YARDAGE: tuple[YardageBucket, ...] = (
    YardageBucket("<120", 110, 12_737 + 13_754, 3 + 6),
    YardageBucket("120-150", 135, 2_936 + 3_798, 2 + 5),
    YardageBucket("150-180", 165, 15_491 + 17_634, 6 + 10),
    YardageBucket("180-200", 190, 15_871 + 17_421, 3 + 8),
    YardageBucket(">200", 215, 23_965 + 21_687, 3 + 13),
)


class MarketQuote(NamedTuple):
    source: str
    prize: float
    golfers: int
    premium: float


def _sheet(source: str, rows: dict[float, tuple[float, float, float]]) -> list[MarketQuote]:
    return [
        MarketQuote(source, prize, golfers, premium)
        for prize, premiums in rows.items()
        for golfers, premium in zip((72, 100, 144), premiums, strict=True)
    ]


MARKET_QUOTES: tuple[MarketQuote, ...] = tuple(
    _sheet(
        "HIOI",
        {
            5_000: (190, 190, 208),
            10_000: (208, 268, 362),
            25_000: (439, 589, 824),
            50_000: (824, 1_123, 1_594),
            60_000: (978, 1_337, 1_901),
        },
    )
    + _sheet(
        "USHIO",
        {
            5_000: (180, 182, 195),
            10_000: (230, 280, 362),
            25_000: (463, 612, 795),
            50_000: (842, 1_048, 1_454),
            100_000: (1_561, 1_976, 2_789),
        },
    )
)

MINIMUM_PREMIUM = 190.0
