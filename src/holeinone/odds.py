"""Per-shot ace odds: odds(y) = odds_150 * exp(beta * (y - 150)).

Amateur: 1 in 12,500 at 150 yards and 1 in 150,000 at 200 (Scheid).
Pro: 1 in 2,500; beta from the PGA Tour fit in calibration.py.
Other classes interpolate beta on log-odds between the two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .frequency import AttemptGroup

REFERENCE_YARDS = 150.0

AMATEUR_ODDS = 12_500.0
AMATEUR_BETA = math.log(150_000 / AMATEUR_ODDS) / 50.0
PRO_ODDS = 2_500.0
PRO_BETA = 0.01166  # fit_yardage_decay(), s.e. 0.0056


@dataclass(frozen=True)
class SkillProfile:
    name: str
    odds_at_reference: float
    beta: float

    def probability(self, yards: float) -> float:
        if yards <= 0:
            raise ValueError("yards must be positive")
        return math.exp(-self.beta * (yards - REFERENCE_YARDS)) / self.odds_at_reference


def interpolated_profile(name: str, odds_at_reference: float) -> SkillProfile:
    t = (math.log(AMATEUR_ODDS) - math.log(odds_at_reference)) / (
        math.log(AMATEUR_ODDS) - math.log(PRO_ODDS)
    )
    t = min(max(t, 0.0), 1.0)
    return SkillProfile(name, odds_at_reference, AMATEUR_BETA + t * (PRO_BETA - AMATEUR_BETA))


SKILLS: dict[str, SkillProfile] = {
    "amateur": SkillProfile("amateur", AMATEUR_ODDS, AMATEUR_BETA),
    "low_handicap": interpolated_profile("low_handicap", 5_000.0),
    "scratch": interpolated_profile("scratch", 3_000.0),
    "pro": SkillProfile("pro", PRO_ODDS, PRO_BETA),
}


def field_groups(
    field: Mapping[str, int],
    yards: float,
    shots_per_golfer: int = 1,
    skills: Mapping[str, SkillProfile] = SKILLS,
) -> list[AttemptGroup]:
    """Attempt groups for a field given as {skill: golfers}."""
    return [
        AttemptGroup(count * shots_per_golfer, skills[name].probability(yards))
        for name, count in field.items()
        if count > 0
    ]
