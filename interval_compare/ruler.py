import math
import statistics
from collections.abc import Sequence

Z80 = statistics.NormalDist().inv_cdf(0.9)


def ruler_band(log_returns: Sequence[float]) -> tuple[float, float, float]:
    if len(log_returns) != 20:
        raise ValueError("ruler needs the last 20 log returns")
    scale = Z80 * statistics.stdev(log_returns) * math.sqrt(3)
    return -scale, scale, 2 * scale
