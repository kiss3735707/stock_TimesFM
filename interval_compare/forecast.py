import math
from collections.abc import Mapping, Sequence

import numpy as np

from interval_compare.decide import bootstrap_deltas, decide
from interval_compare.rows import SnapshotError, row_kind

CHECKPOINT = "google/timesfm-3.0-pytorch"
HORIZON = 3
PREDICT = {
    "return_quantiles": True,
    "use_symmetric_averaging": False,
    "make_positive": False,
    "sort_quantiles": True,
    "use_znorm": False,
    "padding_mode": "none",
}


def three_day_return_quantiles(levels, ln_p_t: float) -> tuple[float, float, float]:
    step = np.asarray(levels)[HORIZON - 1]
    return tuple(float(step[index] - ln_p_t) for index in (0, 4, 8))


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def insufficiency_reason(point: float, ci_low: float, ci_high: float, n: int, min_origins: int = 150) -> str:
    if n < min_origins:
        return "起点少于下限"
    if point >= 0.05 or point <= -0.05:
        return "区间太宽"
    return "点估计没跨过门槛而区间跨过了门槛"


def comparison_from_scores(
    ruler_scores: Sequence[float],
    timesfm_scores: Sequence[float],
    n: int | None = None,
    min_origins: int = 150,
    seed: int = 20261117,
) -> dict:
    if n is None:
        n = len(ruler_scores)
    ruler_mean = sum(ruler_scores) / len(ruler_scores)
    timesfm_mean = sum(timesfm_scores) / len(timesfm_scores)
    point = (ruler_mean - timesfm_mean) / ruler_mean
    replicates = bootstrap_deltas(ruler_scores, timesfm_scores, seed=seed)
    ci_low = percentile(replicates, 2.5)
    ci_high = percentile(replicates, 97.5)
    label = decide(ci_low, ci_high, n, min_origins=min_origins)
    reason = None if label != "证据不足" else insufficiency_reason(point, ci_low, ci_high, n, min_origins)
    return {
        "delta": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "label": label,
        "reason": reason,
        "n": n,
    }


def log_contexts(rows: Sequence[Mapping[str, str]], origins: Sequence[str], context: int) -> list[np.ndarray]:
    normal = [row for row in sorted(rows, key=lambda row: row["date"]) if row_kind(row) == "normal"]
    index = {row["date"]: position for position, row in enumerate(normal)}
    contexts = []
    for origin in origins:
        end = index[origin]
        start = end - context + 1
        if start < 0:
            raise SnapshotError(f"{origin} does not have {context} normal closes")
        window = normal[start : end + 1]
        logs = np.log(np.array([float(row["close"]) for row in window], dtype=np.float64)).astype(np.float32)
        contexts.append(logs)
    return contexts


def forecast_return_bands(forecaster, contexts: Sequence[np.ndarray]) -> list[tuple[float, float, float]]:
    outputs = list(forecaster.predict_batch(contexts=list(contexts), horizon=HORIZON, **PREDICT))
    if len(outputs) != len(contexts):
        raise SnapshotError("forecast count does not match origins")
    bands = []
    for context, output in zip(contexts, outputs):
        if output.quantiles is None:
            raise SnapshotError("predict did not return quantiles")
        bands.append(three_day_return_quantiles(output.quantiles, float(context[-1])))
    return bands


def load_forecaster():
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(CHECKPOINT, device="cpu")


def checkpoint_revision() -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(CHECKPOINT).sha
    except Exception:
        return None
