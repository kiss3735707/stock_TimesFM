import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from interval_compare.qa import check_snapshot
from interval_compare.rows import SnapshotError

CODE = "sh.603232"
FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
    "turn,tradestatus,pctChg,isST"
)


def download_snapshot(out_dir="data", start="2021-01-01", end=None) -> Path:
    end = end or date.today().isoformat()
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise SnapshotError(login.error_msg)
    try:
        basic_fields, basic_rows = _collect(bs.query_stock_basic(code=CODE))
        price_fields, price_rows = _collect(
            bs.query_history_k_data_plus(
                CODE,
                FIELDS,
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="2",
            )
        )
        calendar_fields, calendar_rows = _collect(bs.query_trade_dates(start_date=start, end_date=end))
    finally:
        bs.logout()

    basic = _as_dicts(basic_fields, basic_rows)
    if len(basic) != 1:
        raise SnapshotError("stock basic query did not return one row")
    info = basic[0]
    if info.get("code_name") != "格尔软件":
        raise SnapshotError(f"unexpected name {info.get('code_name')}")
    if info.get("status") != "1" or info.get("outDate"):
        raise SnapshotError("stock is not listed")
    prices = _as_dicts(price_fields, price_rows)
    calendar = _as_dicts(calendar_fields, calendar_rows)
    if not prices or not calendar:
        raise SnapshotError("baostock returned no rows")
    check_snapshot(prices, calendar, name="格尔软件", ipo_date=info.get("ipoDate") or "2017-04-21")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    snapshot = out / f"603232_baostock_{stamp}.csv"
    pd.DataFrame(prices).to_csv(snapshot, index=False)
    pd.DataFrame(calendar).to_csv(out / f"603232_calendar_{stamp}.csv", index=False)
    meta = {
        "name": "格尔软件",
        "code": CODE,
        "ipo_date": info.get("ipoDate") or "2017-04-21",
        "source": "baostock",
        "adjustflag": "2",
        "start": start,
        "end": end,
    }
    (out / f"603232_meta_{stamp}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


def _collect(result) -> tuple[list[str], list[list[str]]]:
    if result.error_code != "0":
        raise SnapshotError(result.error_msg)
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise SnapshotError(result.error_msg)
    return list(result.fields), rows


def _as_dicts(fields: Sequence[str], rows: Sequence[Sequence[str]]) -> list[dict]:
    return [dict(zip(fields, row, strict=True)) for row in rows]
