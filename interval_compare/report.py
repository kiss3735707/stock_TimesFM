import hashlib
import json
import math
import random
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from interval_compare.decide import phase
from interval_compare.qa import check_snapshot
from interval_compare.rows import SnapshotError, row_kind, select_origins
from interval_compare.ruler import ruler_band
from interval_compare.score import interval_score

Row = Mapping[str, str]
MDE_COEFFICIENT = 2.8
SEED = 20261117


def unpaired_mde(scores: Sequence[float], seed: int = SEED, n_boot: int = 1000) -> float:
    if not scores:
        raise ValueError("MDE needs at least one ruler score")
    rng = random.Random(seed)
    n = len(scores)
    replicates = []
    for _ in range(n_boot):
        left = [scores[rng.randrange(n)] for _ in range(n)]
        right = [scores[rng.randrange(n)] for _ in range(n)]
        left_mean = sum(left) / n
        right_mean = sum(right) / n
        if left_mean == 0:
            raise ValueError("ruler mean is zero, so delta is undefined")
        replicates.append((left_mean - right_mean) / left_mean)
    se = float(np.std(replicates, ddof=1))
    return 0.05 + MDE_COEFFICIENT * se


def evaluate_ruler(
    rows: Sequence[Row],
    calendar: Sequence[Row],
    *,
    context: int = 512,
    score_start: str = "2023-11-01",
    prior_returns: int = 20,
    horizon: int = 3,
    seed: int = SEED,
) -> dict:
    if prior_returns != 20:
        raise ValueError("the ruler is defined on 20 log returns")
    ordered = sorted(rows, key=lambda row: row["date"])
    by_date = {row["date"]: row for row in ordered}
    trading = sorted(day["calendar_date"] for day in calendar if day["is_trading_day"] == "1")
    position = {day: index for index, day in enumerate(trading)}
    normal_dates = [row["date"] for row in ordered if row_kind(row) == "normal"]
    normal_count = {day: index + 1 for index, day in enumerate(normal_dates)}
    kept, _ = select_origins(ordered, calendar, prior_returns=prior_returns, horizon=horizon)
    kept_set = set(kept)

    candidates = []
    for day in normal_dates:
        if day < score_start or normal_count[day] < context:
            continue
        if position[day] + horizon >= len(trading):
            continue
        candidates.append(day)
    scored = [day for day in candidates if day in kept_set]
    dropped = [day for day in candidates if day not in kept_set]
    phased = phase(scored)

    scored_records = [_origin_record(day, position, trading, by_date, prior_returns, horizon) for day in scored]
    by_origin = {record["origin"]: record for record in scored_records}
    records = [by_origin[day] for day in phased]

    result = {
        "n_candidates": len(candidates),
        "n_scored": len(scored),
        "n_dropped": len(dropped),
        "drop_rate": (len(dropped) / len(candidates)) if candidates else None,
        "n_phased": len(records),
        "n_st": sum(record["is_st"] for record in records),
        "phased_origins": [record["origin"] for record in records],
        "scored_records": scored_records,
        "records": records,
        "seed": seed,
    }
    if not records:
        result.update(
            mean_score=None,
            coverage=None,
            wilson_low=None,
            wilson_high=None,
            mean_width=None,
            median_abs_error=None,
            median_score=None,
            trimmed_mean_score=None,
            mde=None,
        )
        return result

    scores = [record["score"] for record in records]
    hits = sum(record["hit"] for record in records)
    wilson_low, wilson_high = _wilson(hits, len(records))
    abs_errors = [abs(record["y"]) for record in records]
    result.update(
        mean_score=sum(scores) / len(scores),
        coverage=hits / len(records),
        wilson_low=wilson_low,
        wilson_high=wilson_high,
        mean_width=sum(record["width"] for record in records) / len(records),
        median_abs_error=statistics.median(abs_errors),
        median_score=statistics.median(scores),
        trimmed_mean_score=_trimmed_mean(records),
        mde=unpaired_mde(scores, seed=seed),
    )
    return result


def _origin_record(day, position, trading, by_date, prior_returns, horizon) -> dict:
    index = position[day]
    returns = []
    for step in range(index - prior_returns + 1, index + 1):
        previous = float(by_date[trading[step - 1]]["close"])
        current = float(by_date[trading[step]]["close"])
        returns.append(math.log(current / previous))
    low, high, width = ruler_band(returns)
    future_close = float(by_date[trading[index + horizon]]["close"])
    y = math.log(future_close / float(by_date[day]["close"]))
    return {
        "origin": day,
        "ruler_low": low,
        "ruler_high": high,
        "y": y,
        "hit": int(low <= y <= high),
        "score": interval_score(low, high, y),
        "width": width,
        "is_st": by_date[day].get("isST") == "1",
    }


def render_summary(result: dict) -> str:
    lines = [
        "# 尺子结果",
        "",
        "本里程碑没有 δ，也没有四态结论。",
        "",
        "| 指标 | 值 |",
        "|------|------|",
        f"| 候选起点 | {result['n_candidates']} |",
        f"| 剔除数量 | {result['n_dropped']} |",
        f"| 剔除比例 | {_fmt(result['drop_rate'])} |",
        f"| 不重叠起点 | {result['n_phased']} |",
        f"| ST 起点数量 | {result['n_st']} |",
        f"| 平均区间分数 | {_fmt(result['mean_score'])} |",
        f"| 覆盖率 | {_fmt(result['coverage'])} |",
        f"| Wilson 95% | {_span(result['wilson_low'], result['wilson_high'])} |",
        f"| 平均宽度 | {_fmt(result['mean_width'])} |",
        f"| 中位数绝对误差 | {_fmt(result['median_abs_error'])} |",
        f"| 区间分数中位数 | {_fmt(result['median_score'])} |",
        f"| 去掉绝对收益最大的 5 个起点后的平均区间分数 | {_fmt(result['trimmed_mean_score'])} |",
        f"| MDE | {_fmt(result['mde'])} |",
        "",
        "MDE = 0.05 + 2.8 × SE。SE 是把尺子自己的区间分数独立抽成两份之后，δ 的未配对标准误。",
        "",
    ]
    return "\n".join(lines)


def attach_timesfm(rows, result: dict, forecaster, context: int) -> None:
    from interval_compare.forecast import forecast_return_bands, log_contexts
    from interval_compare.score import interval_score as winkler

    origins = [record["origin"] for record in result["scored_records"]]
    contexts = log_contexts(rows, origins, context)
    bands = forecast_return_bands(forecaster, contexts)
    for record, (low, median, high) in zip(result["scored_records"], bands):
        record["timesfm_low"] = low
        record["timesfm_median"] = median
        record["timesfm_high"] = high
        record["timesfm_hit"] = int(low <= record["y"] <= high)
        record["timesfm_score"] = winkler(low, high, record["y"])
        record["timesfm_width"] = high - low
        record["timesfm_abs_error"] = abs(median - record["y"])
    phased = result["records"]
    from interval_compare.forecast import comparison_from_scores

    result["comparison"] = comparison_from_scores(
        [record["score"] for record in phased],
        [record["timesfm_score"] for record in phased],
        n=len(phased),
        seed=result["seed"],
    )
    result["daily"] = {
        "n": len(result["scored_records"]),
        "ruler": _point_stats(result["scored_records"], "score", "hit", "width", "y"),
        "timesfm": _point_stats(
            result["scored_records"], "timesfm_score", "timesfm_hit", "timesfm_width", "timesfm_abs_error"
        ),
    }


def render_comparison(result: dict) -> str:
    comparison = result["comparison"]
    phased = result["records"]
    timesfm_hits = sum(record["timesfm_hit"] for record in phased)
    timesfm_wilson = _wilson(timesfm_hits, len(phased)) if phased else (None, None)
    reason = f"（{comparison['reason']}）" if comparison["reason"] else ""
    lines = [
        "# 三天区间对照",
        "",
        f"四态结论：{comparison['label']}{reason}",
        "",
        "| 指标 | 尺子 | TimesFM |",
        "|------|------|---------|",
        f"| 平均区间分数 | {_fmt(result['mean_score'])} | {_fmt(_mean(phased, 'timesfm_score'))} |",
        f"| 覆盖率 | {_fmt(result['coverage'])} | {_fmt(timesfm_hits / len(phased) if phased else None)} |",
        f"| Wilson 95% | {_span(result['wilson_low'], result['wilson_high'])} | {_span(*timesfm_wilson)} |",
        f"| 平均宽度 | {_fmt(result['mean_width'])} | {_fmt(_mean(phased, 'timesfm_width'))} |",
        f"| 中位数绝对误差 | {_fmt(result['median_abs_error'])} | {_fmt(_median_key(phased, 'timesfm_abs_error'))} |",
        "",
        "| 判定 | 值 |",
        "|------|------|",
        f"| 不重叠起点 | {result['n_phased']} |",
        f"| δ | {_fmt(comparison['delta'])} |",
        f"| δ 的 95% 区间 | {_span(comparison['ci_low'], comparison['ci_high'])} |",
        f"| 四态 | {comparison['label']} |",
        f"| 证据不足的原因 | {comparison['reason'] or '—'} |",
        f"| 剔除数量 | {result['n_dropped']} |",
        f"| 剔除比例 | {_fmt(result['drop_rate'])} |",
        f"| ST 起点数量 | {result['n_st']} |",
        f"| MDE | {_fmt(result['mde'])} |",
        "",
        "δ =（尺子的平均区间分数 − TimesFM 的平均区间分数）/ 尺子的平均区间分数。分数越低越好。",
        "四态只使用上面的不重叠起点。下面的逐日数字只是点估计，没有置信区间，也不参与四态。",
        "",
    ]
    daily = result.get("daily")
    if daily:
        lines.extend(
            [
                "## 逐日滚动",
                "",
                "| 指标 | 尺子 | TimesFM |",
                "|------|------|---------|",
                f"| 起点数 | {daily['n']} | {daily['n']} |",
                f"| 平均区间分数 | {_fmt(daily['ruler']['mean_score'])} | {_fmt(daily['timesfm']['mean_score'])} |",
                f"| 覆盖率 | {_fmt(daily['ruler']['coverage'])} | {_fmt(daily['timesfm']['coverage'])} |",
                f"| 平均宽度 | {_fmt(daily['ruler']['mean_width'])} | {_fmt(daily['timesfm']['mean_width'])} |",
                f"| 中位数绝对误差 | {_fmt(daily['ruler']['median_abs_error'])} | {_fmt(daily['timesfm']['median_abs_error'])} |",
                "",
            ]
        )
    return "\n".join(lines)


def write_run(snapshot, out, forecaster=None, **kwargs) -> None:
    snapshot_path = Path(snapshot)
    out_path = Path(out)
    calendar_path = _sibling(snapshot_path, "_baostock_", "_calendar_")
    meta_path = _sibling(snapshot_path, "_baostock_", "_meta_").with_suffix(".json")
    rows = _read_table(snapshot_path)
    calendar = _read_table(calendar_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check_snapshot(rows, calendar, name=meta["name"], ipo_date=meta.get("ipo_date", "2017-04-21"))
    result = evaluate_ruler(rows, calendar, **kwargs)
    timesfm = None
    predict = None
    summary = render_summary(result)
    if forecaster is not None:
        from interval_compare.forecast import CHECKPOINT, PREDICT, checkpoint_revision

        attach_timesfm(rows, result, forecaster, context=kwargs.get("context", 512))
        summary = render_comparison(result)
        predict = dict(PREDICT)
        timesfm = {
            "package": _package_version("timesfm"),
            "checkpoint": CHECKPOINT,
            "revision": checkpoint_revision(),
            "device": "cpu",
        }
    out_path.mkdir(parents=True, exist_ok=True)
    _write_rows(out_path / "rows.csv", result["records"])
    (out_path / "summary.md").write_text(summary, encoding="utf-8")
    payload = {
        "git_commit": _git_commit(),
        "snapshot_sha256": _sha256(snapshot_path),
        "calendar_sha256": _sha256(calendar_path),
        "seed": result["seed"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timesfm": timesfm,
        "predict": predict,
        "packages": {
            name: _package_version(name) for name in ("pandas", "numpy", "baostock", "pytest", "timesfm")
        },
    }
    (out_path / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _wilson(hits: int, n: int) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf(0.975)
    proportion = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (proportion + z2 / (2 * n)) / denom
    half = z * math.sqrt(proportion * (1 - proportion) / n + z2 / (4 * n * n)) / denom
    return center - half, center + half


def _trimmed_mean(records: Sequence[dict]) -> float | None:
    if len(records) <= 5:
        return None
    kept = sorted(records, key=lambda record: abs(record["y"]))[:-5]
    return sum(record["score"] for record in kept) / len(kept)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.6f}"


def _span(low, high) -> str:
    if low is None or high is None:
        return "—"
    return f"{low:.6f} – {high:.6f}"


def _sibling(path: Path, old: str, new: str) -> Path:
    return path.with_name(path.name.replace(old, new))


def _read_table(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.to_dict(orient="records")


def _point_stats(records, score_key, hit_key, width_key, error_key) -> dict:
    if not records:
        return {"mean_score": None, "coverage": None, "mean_width": None, "median_abs_error": None}
    errors = [abs(record[error_key]) if error_key == "y" else record[error_key] for record in records]
    return {
        "mean_score": sum(record[score_key] for record in records) / len(records),
        "coverage": sum(record[hit_key] for record in records) / len(records),
        "mean_width": sum(record[width_key] for record in records) / len(records),
        "median_abs_error": statistics.median(errors),
    }


def _mean(records, key):
    if not records:
        return None
    return sum(record[key] for record in records) / len(records)


def _median_key(records, key):
    if not records:
        return None
    return statistics.median(record[key] for record in records)


def _write_rows(path: Path, records: Sequence[dict]) -> None:
    columns = ["origin", "ruler_low", "ruler_high", "y", "hit", "score"]
    if records and "timesfm_low" in records[0]:
        columns += ["timesfm_low", "timesfm_high", "timesfm_hit", "timesfm_score"]
    frame = pd.DataFrame([{key: record[key] for key in columns} for record in records], columns=columns)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if status.strip():
        return revision + "-dirty"
    return revision
