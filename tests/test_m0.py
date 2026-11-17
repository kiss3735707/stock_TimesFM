import math
import statistics

import pytest

from interval_compare.decide import bootstrap_deltas, decide, phase
from interval_compare.rows import (
    SnapshotError,
    filter_scoring_rows,
    is_resume_day,
    select_origins,
)
from interval_compare.ruler import Z80, ruler_band
from interval_compare.score import interval_score


def test_z80_is_the_normal_quantile():
    assert Z80 == statistics.NormalDist().inv_cdf(0.9)
    assert round(Z80, 4) == 1.2816


def test_ruler_band_uses_sample_std_and_sqrt3():
    returns = [0.01, -0.02, 0.015, -0.005] * 5
    scale = Z80 * statistics.stdev(returns) * math.sqrt(3)
    low, high, width = ruler_band(returns)
    assert low == pytest.approx(-scale)
    assert high == pytest.approx(scale)
    assert width == pytest.approx(2 * scale)


def test_ruler_rejects_other_lengths():
    with pytest.raises(ValueError):
        ruler_band([0.01] * 19)


def test_interval_score_penalises_misses_only():
    assert interval_score(-1.0, 1.0, 0.0) == pytest.approx(2.0)
    assert interval_score(-1.0, 1.0, 1.0) == pytest.approx(2.0)
    assert interval_score(-1.0, 1.0, 1.2) == pytest.approx(2.0 + 10 * 0.2)
    assert interval_score(-1.0, 1.0, -1.3) == pytest.approx(2.0 + 10 * 0.3)


def test_four_states_with_floor_injected_as_zero():
    assert decide(0.05, 0.09, n=5, min_origins=0) == "更优"
    assert decide(-0.2, -0.05, n=5, min_origins=0) == "更差"
    assert decide(-0.04, 0.04, n=5, min_origins=0) == "无差异"
    assert decide(0.01, 0.09, n=5, min_origins=0) == "证据不足"


def test_origin_floor_overrides_a_large_effect():
    assert decide(0.5, 0.6, n=149, min_origins=150) == "证据不足"
    assert decide(0.5, 0.6, n=150, min_origins=150) == "更优"


def test_phase_starts_at_the_first_sorted_origin():
    picked = phase(["2024-01-04", "2024-01-01", "2024-01-03", "2024-01-02", "2024-01-05"])
    assert picked == ["2024-01-01", "2024-01-04"]


def test_bootstrap_repeats_are_identical_for_the_locked_seed():
    ruler = [1.0, 1.2, 0.8, 1.1, 0.9, 1.3]
    timesfm = [0.7, 1.4, 0.5, 1.0, 0.6, 1.5]
    first = bootstrap_deltas(ruler, timesfm)
    second = bootstrap_deltas(ruler, timesfm, seed=20261117)
    assert first == second
    assert len(first) == 1000


def test_halt_row_is_dropped():
    rows = [
        _bar("2024-01-02", "1", "100", "1.0"),
        _bar("2024-01-03", "0", "0", ""),
        _bar("2024-01-04", "1", "50", "1.2"),
    ]
    assert [row["date"] for row in filter_scoring_rows(rows)] == ["2024-01-02", "2024-01-04"]


def test_contradictory_status_and_volume_stops():
    with pytest.raises(SnapshotError):
        filter_scoring_rows([_bar("2024-01-02", "1", "0", "")])


def test_resume_day_is_kept():
    rows = [
        _bar("2024-01-02", "0", "0", ""),
        _bar("2024-01-03", "1", "80", "2.0"),
    ]
    assert is_resume_day(rows, 1) is True
    assert [row["date"] for row in filter_scoring_rows(rows)] == ["2024-01-03"]


def test_first_snapshot_row_is_not_a_resume_day():
    rows = [_bar("2024-01-02", "1", "80", "2.0")]
    assert is_resume_day(rows, 0) is False


def test_missing_trading_day_drops_that_origin():
    days = [f"2024-01-0{n}" for n in range(1, 7)]
    calendar = [{"calendar_date": day, "is_trading_day": "1"} for day in days]
    calendar.append({"calendar_date": "2023-12-31", "is_trading_day": "0"})

    def snapshot(missing=None):
        return [_bar(day, "1", "10", "1") for day in days if day != missing]

    kept, dropped = select_origins(snapshot(), calendar, prior_returns=2, horizon=3)
    assert kept == ["2024-01-03"]
    assert dropped == []

    kept_gap, dropped_gap = select_origins(snapshot("2024-01-05"), calendar, prior_returns=2, horizon=3)
    assert "2024-01-03" not in kept_gap
    assert dropped_gap == ["2024-01-03"]
    assert len(dropped_gap) == len(dropped) + 1


def _bar(date, status, volume, turn):
    return {"date": date, "tradestatus": status, "volume": volume, "turn": turn}
