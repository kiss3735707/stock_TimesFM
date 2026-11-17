def interval_score(low: float, high: float, y: float) -> float:
    if low > high:
        raise ValueError("interval lower bound is above the upper bound")
    width = high - low
    miss = max(low - y, 0.0) + max(y - high, 0.0)
    return width + (2 / 0.2) * miss
