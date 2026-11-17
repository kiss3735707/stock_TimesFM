from collections.abc import Mapping, Sequence

from interval_compare.rows import SnapshotError, is_resume_day, row_kind

Row = Mapping[str, str]
PRICE_FIELDS = ("open", "high", "low", "close", "preclose")
LIMIT = 10.1


def check_snapshot(
    rows: Sequence[Row],
    calendar: Sequence[Row],
    *,
    name: str,
    ipo_date: str = "2017-04-21",
) -> None:
    if name != "格尔软件":
        raise SnapshotError(f"unexpected name {name}")
    ordered = sorted(rows, key=lambda row: row["date"])
    seen: set[str] = set()
    for index, row in enumerate(ordered):
        if row.get("code") != "sh.603232":
            raise SnapshotError(f"unexpected code on {row.get('date')}")
        if row.get("adjustflag") != "2":
            raise SnapshotError(f"adjustflag is not 2 on {row.get('date')}")
        if row["date"] in seen:
            raise SnapshotError(f"duplicate date {row['date']}")
        seen.add(row["date"])
        for field in PRICE_FIELDS:
            _positive(row, field)
        kind = row_kind(row)
        if _exempt_from_limit(ordered, index, row, ipo_date):
            continue
        if kind == "halt":
            continue
        _check_limit(row)
    _check_calendar(seen, calendar)


def _positive(row: Row, field: str) -> None:
    raw = row.get(field, "")
    if raw == "":
        raise SnapshotError(f"{field} is empty on {row.get('date')}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SnapshotError(f"{field} is not a number on {row.get('date')}") from exc
    if value <= 0:
        raise SnapshotError(f"{field} is not positive on {row.get('date')}")


def _exempt_from_limit(rows: Sequence[Row], index: int, row: Row, ipo_date: str) -> bool:
    if row.get("isST") == "1":
        return True
    if row["date"] == ipo_date:
        return True
    return is_resume_day(rows, index)


def _check_limit(row: Row) -> None:
    raw = row.get("pctChg", "")
    if raw == "":
        raise SnapshotError(f"pctChg is empty on {row['date']}")
    try:
        move = abs(float(raw))
    except ValueError as exc:
        raise SnapshotError(f"pctChg is not a number on {row['date']}") from exc
    if move > LIMIT:
        raise SnapshotError(f"pctChg {raw} is outside ±{LIMIT} on {row['date']}")


def _check_calendar(dates: set[str], calendar: Sequence[Row]) -> None:
    if not dates:
        raise SnapshotError("snapshot has no rows")
    start, end = min(dates), max(dates)
    trading = {
        day["calendar_date"]
        for day in calendar
        if day["is_trading_day"] == "1" and start <= day["calendar_date"] <= end
    }
    missing = sorted(trading - dates)
    if missing:
        raise SnapshotError(f"missing trading day {missing[0]}")
    extra = sorted(dates - trading)
    if extra:
        raise SnapshotError(f"price row is not a trading day {extra[0]}")
