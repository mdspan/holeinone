import pytest

from holeinone import odds


def test_amateur_anchors():
    am = odds.SKILLS["amateur"]
    assert 1 / am.probability(150) == pytest.approx(12_500)
    assert 1 / am.probability(200) == pytest.approx(150_000)


def test_skills_are_ordered_at_every_length():
    for y in range(110, 231, 10):
        p = [s.probability(y) for s in odds.SKILLS.values()]
        if y >= odds.REFERENCE_YARDS:
            assert p == sorted(p)


def test_field_groups():
    groups = odds.field_groups({"amateur": 100, "pro": 0, "scratch": 4}, 165, shots_per_golfer=2)
    assert [g.shots for g in groups] == [200, 8]
    assert groups[0].p == pytest.approx(odds.SKILLS["amateur"].probability(165))
