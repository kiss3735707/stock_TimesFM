import random
from collections.abc import Sequence


def phase(dates: Sequence[str]) -> list[str]:
    ordered = sorted(dates)
    return ordered[::3]


def bootstrap_deltas(
    ruler_scores: Sequence[float],
    timesfm_scores: Sequence[float],
    seed: int = 20261117,
    n_boot: int = 1000,
) -> list[float]:
    if len(ruler_scores) != len(timesfm_scores) or not ruler_scores:
        raise ValueError("the two score series must be the same non-empty length")
    rng = random.Random(seed)
    n = len(ruler_scores)
    replicates = []
    for _ in range(n_boot):
        drawn = [rng.randrange(n) for _ in range(n)]
        ruler_mean = sum(ruler_scores[i] for i in drawn) / n
        timesfm_mean = sum(timesfm_scores[i] for i in drawn) / n
        if ruler_mean == 0:
            raise ValueError("ruler mean is zero, so delta is undefined")
        replicates.append((ruler_mean - timesfm_mean) / ruler_mean)
    return replicates


def decide(ci_low: float, ci_high: float, n: int, min_origins: int = 150) -> str:
    if n < min_origins:
        return "证据不足"
    if ci_low >= 0.05:
        return "更优"
    if ci_high <= -0.05:
        return "更差"
    if ci_low > -0.05 and ci_high < 0.05:
        return "无差异"
    return "证据不足"
