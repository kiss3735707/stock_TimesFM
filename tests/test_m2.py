import json
from types import SimpleNamespace

import numpy as np
import pytest

from interval_compare.forecast import (
    PREDICT,
    comparison_from_scores,
    insufficiency_reason,
    log_contexts,
    percentile,
    three_day_return_quantiles,
)
from interval_compare.report import write_run
from tests.test_m1 import _calendar, _smooth_history, _write_csv


def test_three_day_quantiles_subtract_log_price_at_t():
    levels = np.zeros((3, 9))
    levels[2, 0] = 1.1
    levels[2, 4] = 1.2
    levels[2, 8] = 1.4
    low, median, high = three_day_return_quantiles(levels, ln_p_t=1.0)
    assert (low, median, high) == pytest.approx((0.1, 0.2, 0.4))
    assert low + median + high != pytest.approx(1.1 + 1.2 + 1.4)


def test_percentile_is_linear():
    assert percentile([0, 10], 2.5) == pytest.approx(0.25)


def test_constant_improvement_is_better_when_floor_is_zero():
    result = comparison_from_scores([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], min_origins=0)
    assert result["delta"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)
    assert result["label"] == "更优"
    assert result["reason"] is None


def test_insufficiency_reasons():
    assert insufficiency_reason(0.5, 0.4, 0.6, n=149) == "起点少于下限"
    assert insufficiency_reason(0.08, 0.01, 0.09, n=200) == "区间太宽"
    assert insufficiency_reason(0.02, -0.08, 0.10, n=200) == "点估计没跨过门槛而区间跨过了门槛"


def test_log_context_is_the_last_512_normal_closes():
    rows, _calendar_unused = _smooth_history(6)
    contexts = log_contexts(rows, ["2024-01-08"], context=4)
    assert len(contexts) == 1
    assert contexts[0].shape == (4,)
    assert contexts[0].dtype == np.float32
    assert float(contexts[0][-1]) == pytest.approx(np.log(float(rows[-1]["close"])))


def test_run_with_forecaster_writes_the_four_state_call(tmp_path):
    rows, calendar = _smooth_history(24)
    snapshot = tmp_path / "603232_baostock_20241117.csv"
    _write_csv(snapshot, rows)
    _write_csv(tmp_path / "603232_calendar_20241117.csv", calendar)
    (tmp_path / "603232_meta_20241117.json").write_text(
        json.dumps({"name": "格尔软件", "ipo_date": "2017-04-21"}),
        encoding="utf-8",
    )
    forecaster = _FakeForecaster()
    out = tmp_path / "out"
    write_run(snapshot, out, forecaster=forecaster, context=21, score_start="2024-01-01")
    assert forecaster.kwargs == PREDICT
    assert forecaster.horizon == 3
    assert "device" not in forecaster.kwargs
    assert len(forecaster.contexts[0]) == 21
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "四态" in text
    assert "证据不足" in text
    assert "起点少于下限" in text
    header = (out / "rows.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "timesfm_low" in header
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert payload["predict"]["return_quantiles"] is True
    assert payload["predict"]["padding_mode"] == "none"


class _FakeForecaster:
    def predict_batch(self, contexts, horizon, **kwargs):
        self.contexts = contexts
        self.horizon = horizon
        self.kwargs = kwargs
        outputs = []
        for context in contexts:
            quantiles = np.zeros((horizon, 9), dtype=np.float32)
            last = float(context[-1])
            quantiles[2, 0] = last - 0.2
            quantiles[2, 4] = last
            quantiles[2, 8] = last + 0.2
            outputs.append(SimpleNamespace(quantiles=quantiles))
        return outputs
