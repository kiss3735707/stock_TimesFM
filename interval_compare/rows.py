from collections.abc import Mapping, Sequence

SnapshotRow = Mapping[str, str]


class SnapshotError(ValueError):
    pass


def row_kind(row: SnapshotRow) -> str:
    status = row["tradestatus"]
    volume = float(row["volume"])
    if status == "1" and volume > 0:
        return "normal"
    if status == "0" and volume == 0:
        return "halt"
    raise SnapshotError("tradestatus and volume disagree")


def filter_scoring_rows(rows: Sequence[SnapshotRow]) -> list[SnapshotRow]:
    return [row for row in rows if row_kind(row) == "normal"]


def is_resume_day(rows: Sequence[SnapshotRow], index: int) -> bool:
    if index <= 0:
        return False
    return rows[index - 1]["tradestatus"] == "0" and rows[index]["tradestatus"] == "1"


def select_origins(
    rows: Sequence[SnapshotRow],
    calendar: Sequence[Mapping[str, str]],
    prior_returns: int = 20,
    horizon: int = 3,
) -> tuple[list[str], list[str]]:
    by_date: dict[str, SnapshotRow] = {}
    for row in rows:
        row_kind(row)
        date = row["date"]
        if date in by_date:
            raise SnapshotError(f"duplicate date {date}")
        by_date[date] = row

    trading = sorted(
        day["calendar_date"] for day in calendar if day["is_trading_day"] == "1"
    )
    kept: list[str] = []
    dropped: list[str] = []
    for index, day in enumerate(trading):
        if index < prior_returns or index + horizon >= len(trading):
            continue
        if day not in by_date or row_kind(by_date[day]) != "normal":
            continue
        window = trading[index - prior_returns : index + horizon + 1]
        if all(_is_normal(by_date.get(date)) for date in window):
            kept.append(day)
        else:
            dropped.append(day)
    return kept, dropped


def _is_normal(row: SnapshotRow | None) -> bool:
    return row is not None and row_kind(row) == "normal"
