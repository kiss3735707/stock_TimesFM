import csv
import json
import math
from datetime import date, timedelta

import pytest

from interval_compare.qa import check_snapshot
from interval_compare.report import evaluate_ruler, render_summary, unpaired_mde, write_run
from interval_compare.rows import SnapshotError
from interval_compare.score import interval_score


def test_pctchg_above_10_1_stops():
    rows = [_price("2024-01-02", pct="10.11")]
    with pytest.raises(SnapshotError):
        check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_pctchg_at_10_1_passes():
    rows = [_price("2024-01-02", pct="10.1")]
    check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_resume_day_skips_the_limit_check():
    rows = [
        _price("2024-01-02", status="0", volume="0", turn="", pct=""),
        _price("2024-01-03", pct="20"),
    ]
    check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_first_row_is_not_exempt_from_the_limit():
    rows = [_price("2024-01-02", pct="12")]
    with pytest.raises(SnapshotError):
        check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_listing_day_skips_the_limit_check():
    rows = [_price("2017-04-21", pct="44")]
    check_snapshot(rows, _calendar(rows), name="格尔软件", ipo_date="2017-04-21")


def test_st_row_is_not_stopped_for_a_wide_move():
    rows = [_price("2024-01-02", pct="30", is_st="1")]
    check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_halt_may_have_empty_turn_and_pctchg():
    rows = [
        _price("2024-01-02"),
        _price("2024-01-03", status="0", volume="0", turn="", pct=""),
        _price("2024-01-04"),
    ]
    check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_missing_trading_day_stops():
    rows = [_price("2024-01-02"), _price("2024-01-04")]
    calendar = [
        {"calendar_date": "2024-01-02", "is_trading_day": "1"},
        {"calendar_date": "2024-01-03", "is_trading_day": "1"},
        {"calendar_date": "2024-01-04", "is_trading_day": "1"},
    ]
    with pytest.raises(SnapshotError):
        check_snapshot(rows, calendar, name="格尔软件")


def test_non_positive_close_stops():
    rows = [_price("2024-01-02", close="0")]
    with pytest.raises(SnapshotError):
        check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_wrong_code_stops():
    rows = [_price("2024-01-02", code="sz.000001")]
    with pytest.raises(SnapshotError):
        check_snapshot(rows, _calendar(rows), name="格尔软件")


def test_constant_scores_have_mde_floor():
    assert unpaired_mde([0.2, 0.2, 0.2]) == pytest.approx(0.05)


def test_mde_repeats_for_the_locked_seed():
    scores = [0.2, 0.5, 0.1, 0.4, 0.3]
    assert unpaired_mde(scores) == unpaired_mde(scores)


def test_ruler_report_on_one_origin_and_no_delta():
    rows, calendar = _smooth_history(24)
    result = evaluate_ruler(rows, calendar, context=21, score_start="2024-01-01")
    y = 0.003
    assert result["n_phased"] == 1
    assert result["n_dropped"] == 0
    assert result["coverage"] == 0
    assert result["mean_width"] == pytest.approx(0)
    assert result["median_abs_error"] == pytest.approx(y)
    assert result["mean_score"] == pytest.approx(interval_score(0.0, 0.0, y))
    assert result["mde"] == pytest.approx(0.05)
    text = render_summary(result)
    assert "没有 δ" in text
    assert "更优" not in text
    assert f"{result['mde']:.6f}" in text


def test_phase_keeps_the_first_scored_origin():
    rows, calendar = _smooth_history(26)
    result = evaluate_ruler(rows, calendar, context=21, score_start="2024-01-01")
    assert result["n_scored"] == 3
    assert result["n_phased"] == 1
    assert result["phased_origins"] == ["2024-01-29"]


def test_run_writes_ruler_summary_without_a_four_state_call(tmp_path):
    rows, calendar = _smooth_history(24)
    snapshot = tmp_path / "603232_baostock_20241117.csv"
    _write_csv(snapshot, rows)
    _write_csv(tmp_path / "603232_calendar_20241117.csv", calendar)
    (tmp_path / "603232_meta_20241117.json").write_text(
        json.dumps({"name": "格尔软件", "ipo_date": "2017-04-21"}),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    write_run(snapshot, out, context=21, score_start="2024-01-01")
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "没有 δ" in text
    assert "更优" not in text
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert payload["timesfm"] is None
    assert payload["seed"] == 20261117
    assert "2024-01-29" in (out / "rows.csv").read_text(encoding="utf-8")


def test_halt_inside_the_window_counts_as_a_drop():
    rows, calendar = _smooth_history(25)
    halted = "2024-01-29"
    for row in rows:
        if row["date"] == halted:
            row["tradestatus"] = "0"
            row["volume"] = "0"
            row["turn"] = ""
            row["pctChg"] = ""
    result = evaluate_ruler(rows, calendar, context=21, score_start="2024-01-01")
    assert result["n_candidates"] == 1
    assert result["n_dropped"] == 1
    assert result["n_phased"] == 0
    assert result["drop_rate"] == pytest.approx(1)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _price(
    day,
    *,
    code="sh.603232",
    status="1",
    volume="100",
    turn="1",
    pct="1.0",
    close="10",
    is_st="0",
):
    return {
        "date": day,
        "code": code,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "preclose": close,
        "volume": volume,
        "amount": "1000",
        "adjustflag": "2",
        "turn": turn,
        "tradestatus": status,
        "pctChg": pct,
        "isST": is_st,
    }


def _calendar(rows):
    days = sorted(row["date"] for row in rows)
    start = date.fromisoformat(days[0])
    end = date.fromisoformat(days[-1])
    out = []
    cursor = start
    trading = set(days)
    while cursor <= end:
        iso = cursor.isoformat()
        out.append({"calendar_date": iso, "is_trading_day": "1" if iso in trading else "0"})
        cursor += timedelta(days=1)
    return out


def _smooth_history(n):
    start = date(2024, 1, 1)
    rows = []
    day = start
    while len(rows) < n:
        if day.weekday() < 5:
            close = str(10 * math.exp(0.001 * len(rows)))
            rows.append(_price(day.isoformat(), close=close, pct="0.1"))
        day += timedelta(days=1)
    return rows, _calendar(rows)
