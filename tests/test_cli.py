import pytest

from holeinone.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["quote", "--prize", "25000", "--golfers", "144"],
        [
            "quote",
            "--prize",
            "1000000",
            "--golfers",
            "6",
            "--annuity-years",
            "40",
            "--discount-rate",
            "0.04",
            "--basis",
            "each",
        ],
        ["odds", "--step", "20"],
        ["market"],
        ["simulate", "--contests", "20", "--sims", "2000", "--quota-share", "0.5"],
    ],
)
def test_cli_runs(argv, capsys):
    main(argv)
    assert capsys.readouterr().out.strip()
