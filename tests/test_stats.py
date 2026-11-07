import numpy as np
import pandas as pd
import pytest

from analysis import stats as S
from analysis.windows import WindowError, parse_duration, parse_window


def sd(values, index, **kw):
    return S.prepare_series(pd.Series(values, index=index), metric=kw.pop("metric", "v"), **kw)


def test_descriptive_known_values(ts_index):
    r = S.descriptive(sd([1.0, 2.0, 3.0, 4.0], ts_index(4)))
    assert r["mean"] == 2.5 and r["median"] == 2.5
    assert r["min"] == 1.0 and r["max"] == 4.0
    assert r["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert r["change"] == 3.0 and r["change_pct"] == pytest.approx(300.0)
    assert r["span_seconds"] == 6.0
    assert r["provenance"]["n"] == 4


def test_percentage_change_zero_and_missing():
    assert S.percentage_change(0, 5)["value"] is None
    assert "zero" in S.percentage_change(0, 5)["reason"]
    assert S.percentage_change(None, 5)["value"] is None
    assert S.percentage_change(-10, -5)["value"] == pytest.approx(50.0)
    assert S.percentage_change(200, 150)["value"] == pytest.approx(-25.0)


def test_rate_uses_actual_irregular_timestamps():
    # value = 2 × minutes elapsed, sampled at irregular instants -> slope must be exactly 2 per minute
    secs = np.array([0, 7, 9, 40, 41, 95, 180, 181, 300])
    idx = pd.Timestamp("2026-10-29T10:00Z") + pd.to_timedelta(secs, unit="s")
    r = S.rate_of_change(sd(2 * secs / 60.0, idx, unit="kW"))
    assert r["slope"] == pytest.approx(2.0, rel=1e-9)
    assert r["endpoint_rate"] == pytest.approx(2.0, rel=1e-9)
    assert r["rate_unit"] == "kW per minute"


def test_rate_timestamp_resolution_independent():
    # Regression: pandas 3 may store datetimes in microseconds; rates must not be off by 1000x.
    idx_us = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=11, freq="1min", tz="UTC").as_unit("us"))
    idx_ns = idx_us.as_unit("ns")
    vals = np.arange(11) * 5.0
    assert S.rate_of_change(sd(vals, idx_us))["slope"] == pytest.approx(5.0)
    assert S.rate_of_change(sd(vals, idx_ns))["slope"] == pytest.approx(5.0)


def test_prepare_handles_duplicates_nan_nonnumeric_unsorted():
    idx = pd.to_datetime(["2026-01-01 00:00:02", "2026-01-01 00:00:00", "2026-01-01 00:00:02", "2026-01-01 00:00:04", "2026-01-01 00:00:06"], utc=True)
    s = S.prepare_series(pd.Series([4.0, 1.0, 6.0, "x", np.inf], index=idx), "m")
    assert s.n == 2  # 00:00 and the averaged 00:02
    assert s.series.iloc[1] == 5.0
    joined = " ".join(s.notes)
    assert "duplicate" in joined and "sorted" in joined and "excluded" in joined


def test_insufficient_data_raises(ts_index):
    with pytest.raises(S.InsufficientData):
        S.rate_of_change(sd([1.0], ts_index(1)))
    with pytest.raises(S.InsufficientData):
        S.trend(sd([1.0, 2.0], ts_index(2)))
    with pytest.raises(S.InsufficientData):
        S.rate_of_change(sd([1.0, 2.0], pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")] * 2)))


def test_trend_detects_increase_and_flat(ts_index):
    rng = np.random.default_rng(0)
    up = S.trend(sd(np.arange(50) * 0.5 + rng.normal(0, 0.1, 50), ts_index(50)))
    assert up["direction"] == "increasing" and up["significant"] and up["p_value"] < 0.001
    flat = S.trend(sd(np.ones(20), ts_index(20)))
    assert flat["direction"] == "flat"
    noise = S.trend(sd(np.random.default_rng(3).normal(0, 1, 30), ts_index(30)))
    assert noise["direction"] in ("no significant trend", "increasing", "decreasing")


def test_anomalies_find_injected_spike_and_none_for_constant(ts_index):
    rng = np.random.default_rng(1)
    v = 230 + rng.normal(0, 0.5, 200)
    v[120] = 260
    r = S.detect_anomalies(sd(v, ts_index(200)))
    assert r["count"] >= 1
    assert any(a["value"] == pytest.approx(260) and a["direction"] == "high" for a in r["anomalies"])
    assert S.detect_anomalies(sd(np.full(20, 5.0), ts_index(20)))["count"] == 0


def test_peaks(ts_index):
    v = np.zeros(50)
    v[10], v[30] = 10, 8
    r = S.detect_peaks(sd(v, ts_index(50)))
    assert [p["value"] for p in r["peaks"]] == [10, 8]


def test_threshold_violations_with_persistence(ts_index):
    v = [1, 1, 5, 5, 5, 1, 5, 1]
    r = S.threshold_violations(sd(v, ts_index(8, "1s")), upper=3, min_duration_s=2)
    assert r["count"] == 1
    assert r["episodes"][0]["duration_s"] == 2 and r["episodes"][0]["samples"] == 3
    assert S.threshold_violations(sd(v, ts_index(8, "1s")), upper=3)["count"] == 2
    with pytest.raises(ValueError):
        S.threshold_violations(sd(v, ts_index(8)), None, None)


def test_compare_windows(ts_index):
    a = sd([1.0] * 10, ts_index(10))
    b = sd([2.0] * 10, ts_index(10, start="2026-10-29 11:00"))
    r = S.compare_windows(a, b, "before", "after")
    assert r["before"]["mean"] == 1 and r["after"]["mean"] == 2
    assert r["mean_change_pct"] == pytest.approx(100.0)
    assert r["direction"] == "higher"
    z = S.compare_windows(sd([0.0] * 5, ts_index(5)), sd([1.0] * 5, ts_index(5)), "a", "b")
    assert z["mean_change_pct"] is None and "zero" in z["mean_change_note"]


def test_correlation_and_alignment(ts_index):
    idx = ts_index(100)
    x = np.linspace(0, 10, 100)
    a = sd(x, idx, metric="a")
    b = sd(3 * x + 1, idx + pd.Timedelta(milliseconds=300), metric="b")  # slightly offset clocks
    r = S.correlation(a, b)
    assert r["paired_samples"] == 100
    assert r["pearson_r"] == pytest.approx(1.0)
    assert "not establish" in r["caution"]
    const = S.correlation(a, sd(np.ones(100), idx, metric="c"))
    assert const["pearson_r"] is None


def test_rank_rates_relative(ts_index):
    idx = ts_index(30, "1min")
    slow = sd(1000 + np.arange(30) * 1.0, idx, metric="big", unit="W")  # 0.1 %/min
    fast = sd(10 + np.arange(30) * 1.0, idx, metric="small", unit="A")  # ~4 %/min
    r = S.rank_rates([slow, fast, sd([1.0], idx[:1], metric="one")])
    assert [x["metric"] for x in r["ranking"]] == ["small", "big"]
    assert r["skipped"][0]["metric"] == "one"


def test_row_index_basis_reports_per_row(ts_index):
    s = sd([1.0, 2.0, 3.0], ts_index(3), time_basis="row_index")
    assert S.rate_of_change(s)["rate_unit"].endswith("per row")
    assert any("row order" in n for n in s.notes)


def test_quality_score(ts_index):
    idx = ts_index(10, "1s").delete([3, 4, 5])
    q = S.data_quality(sd(np.arange(7.0), idx), expected_interval_s=1)
    assert q["completeness"] == pytest.approx(0.7)
    assert q["largest_gap_s"] == 4


def test_moving_average(ts_index):
    s = sd([1.0, 2.0, 3.0, 4.0], ts_index(4, "1s"))
    assert S.moving_average(s, 2).tolist() == [1.0, 1.5, 2.5, 3.5]
    assert S.moving_average(s, "2s").iloc[-1] == 3.5


# --------------------------------------------------------------- windows
def test_parse_duration_variants():
    assert parse_duration("last 10 minutes").total_seconds() == 600
    assert parse_duration("previous ten minutes").total_seconds() == 600
    assert parse_duration("past hour").total_seconds() == 3600
    assert parse_duration("15m").total_seconds() == 900
    assert parse_duration("2 h").total_seconds() == 7200
    assert parse_duration("nothing here") is None


def test_parse_window_anchors():
    now = pd.Timestamp("2026-10-29T12:00Z").to_pydatetime()
    w = parse_window("last 10 minutes", now=now)
    assert w.anchor == "now" and w.seconds == 600 and w.end == now
    prev = parse_window("previous 10 minutes", now=now)
    assert prev.end == w.start and prev.seconds == 600
    old = pd.Timestamp("2026-01-01T00:00Z").to_pydatetime()
    h = parse_window("last 10 minutes", data_end=old, now=now)
    assert h.anchor == "latest-data" and h.end == old
    r = parse_window("2026-01-01T00:00/2026-01-01T01:00", now=now)
    assert r.seconds == 3600 and r.anchor == "explicit"
    with pytest.raises(WindowError):
        parse_window("sometime", now=now)
    with pytest.raises(WindowError):
        parse_window("2026-01-01T01:00/2026-01-01T00:00", now=now)
