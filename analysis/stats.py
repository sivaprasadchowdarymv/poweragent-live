"""Deterministic numerical analysis on timestamped series.

Every public function takes a ``pd.Series`` indexed by UTC timestamps (or a
prepared ``SeriesData``) and returns a JSON-serializable dict. Numbers are
never produced by the language model. Rates use the *actual* measurement
timestamps, never an assumed capture interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, stats

MIN_FOR_STATS = 1
MIN_FOR_TREND = 3


class InsufficientData(ValueError):
    pass


@dataclass
class SeriesData:
    """A cleaned series plus everything needed to reproduce a result."""

    series: pd.Series
    metric: str
    unit: str = ""
    source: str = ""
    status: str = "verified"  # verified | estimated | mixed
    time_basis: str = "timestamp"  # or "row_index"
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.series.size)

    def provenance(self) -> dict[str, Any]:
        s = self.series
        return {
            "metric": self.metric,
            "unit": self.unit,
            "source": self.source,
            "status": self.status,
            "time_basis": self.time_basis,
            "n": self.n,
            "first_timestamp": s.index[0].isoformat() if self.n else None,
            "last_timestamp": s.index[-1].isoformat() if self.n else None,
            "notes": list(self.notes),
        }


def epoch_seconds(idx: Any) -> np.ndarray:
    """Seconds since the Unix epoch for a DatetimeIndex/Series, independent of the
    datetime resolution (pandas 3 may store us or ns)."""
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        di = di.tz_localize("UTC")
    return np.asarray((di - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds(), dtype=float)


def finite(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def prepare_series(
    values: pd.Series | pd.DataFrame,
    metric: str = "",
    unit: str = "",
    source: str = "",
    status: str = "verified",
    time_basis: str = "timestamp",
) -> SeriesData:
    """Clean a series: numeric coercion, drop NaN/inf, sort by time, average
    duplicate timestamps (reported, not hidden), flag irregular sampling."""
    if isinstance(values, pd.DataFrame):
        s = pd.Series(values["value"].values, index=pd.to_datetime(values["timestamp"], utc=True))
    else:
        s = values.copy()
    notes: list[str] = []
    s = pd.to_numeric(s, errors="coerce")
    bad = int(s.isna().sum())
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if bad:
        notes.append(f"{bad} missing or non-numeric value(s) were excluded.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Series must be indexed by timestamps.")
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    # One datetime resolution everywhere: sources parsed differently (s/us/ns) must still align.
    s.index = s.index.tz_convert("UTC").as_unit("ns")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index(kind="stable")
        notes.append("Values were sorted by timestamp.")
    dup = int(s.index.duplicated().sum())
    if dup:
        s = s.groupby(level=0).mean()
        notes.append(f"{dup} duplicate timestamp(s) were averaged.")
    if s.size >= 3:
        dt = np.diff(epoch_seconds(s.index))
        med = float(np.median(dt))
        if med > 0 and (np.max(dt) > 3 * med):
            gaps = int(np.sum(dt > 3 * med))
            notes.append(f"Sampling is irregular: {gaps} gap(s) longer than 3× the median interval ({med:.3g} s).")
    if time_basis == "row_index":
        notes.append("This source has no timestamp column; 'time' is the row order, so rates are per row.")
    return SeriesData(s.astype(float), metric, unit, source, status, time_basis, notes)


def _require(sd: SeriesData, n: int, what: str) -> None:
    if sd.n < n:
        raise InsufficientData(f"{what} needs at least {n} valid observation(s) of '{sd.metric}'; found {sd.n}.")


def _ts(t: pd.Timestamp) -> str:
    return t.isoformat()


def _rate_unit(sd: SeriesData) -> tuple[float, str]:
    if sd.time_basis == "row_index":
        return 1.0, "per row"
    return 60.0, "per minute"


# ------------------------------------------------------------------ basics
def percentage_change(old: float | None, new: float | None) -> dict[str, Any]:
    """(new - old) / |old| × 100. A zero or missing baseline yields ``None`` with a reason."""
    o, n = finite(old), finite(new)
    if o is None or n is None:
        return {"value": None, "absolute_change": None, "reason": "A value is missing."}
    if o == 0:
        return {"value": None, "absolute_change": n - o, "reason": "The baseline is zero, so a percentage change is undefined."}
    return {"value": (n - o) / abs(o) * 100.0, "absolute_change": n - o, "reason": ""}


def descriptive(sd: SeriesData) -> dict[str, Any]:
    _require(sd, MIN_FOR_STATS, "Statistics")
    s = sd.series
    imax, imin = s.idxmax(), s.idxmin()
    span = (s.index[-1] - s.index[0]).total_seconds()
    pc = percentage_change(s.iloc[0], s.iloc[-1])
    return {
        "count": sd.n,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if sd.n > 1 else 0.0,
        "min": float(s.min()),
        "min_at": _ts(imin),
        "max": float(s.max()),
        "max_at": _ts(imax),
        "range": float(s.max() - s.min()),
        "first": float(s.iloc[0]),
        "first_at": _ts(s.index[0]),
        "last": float(s.iloc[-1]),
        "last_at": _ts(s.index[-1]),
        "change": float(s.iloc[-1] - s.iloc[0]),
        "change_pct": pc["value"],
        "change_pct_note": pc["reason"],
        "span_seconds": span if sd.time_basis == "timestamp" else None,
        "provenance": sd.provenance(),
    }


def moving_average(sd: SeriesData, window: str | int = "60s") -> pd.Series:
    """Time-based (e.g. '60s', '5min') or sample-count rolling mean."""
    if isinstance(window, int):
        if window < 1:
            raise ValueError("Window must be at least 1 sample.")
        return sd.series.rolling(window, min_periods=1).mean()
    return sd.series.rolling(pd.Timedelta(window), min_periods=1).mean()


def rate_of_change(sd: SeriesData) -> dict[str, Any]:
    """Least-squares slope against real timestamps, plus the endpoint rate."""
    _require(sd, 2, "A rate of change")
    s = sd.series
    t = epoch_seconds(s.index) - epoch_seconds(s.index)[0]
    if sd.time_basis == "row_index":
        t = np.arange(sd.n, dtype=float)
    span = float(t[-1] - t[0])
    if span <= 0:
        raise InsufficientData(f"All observations of '{sd.metric}' share one timestamp; a rate cannot be computed.")
    mult, label = _rate_unit(sd)
    slope = float(np.polyfit(t, s.values, 1)[0]) if sd.n >= 3 else float((s.iloc[-1] - s.iloc[0]) / span)
    endpoint = float((s.iloc[-1] - s.iloc[0]) / span)
    level = float(np.mean(np.abs(s.values)))
    rel = slope * mult / level * 100 if level > 0 else None
    return {
        "slope": slope * mult,
        "endpoint_rate": endpoint * mult,
        "rate_unit": f"{sd.unit + ' ' if sd.unit else ''}{label}".strip(),
        "relative_rate_pct": rel,
        "relative_rate_unit": "% of mean level " + label,
        "span": span,
        "provenance": sd.provenance(),
    }


def trend(sd: SeriesData, alpha: float = 0.05) -> dict[str, Any]:
    """Linear fit (slope, R²) and Kendall's tau for a monotonic-trend test."""
    _require(sd, MIN_FOR_TREND, "Trend analysis")
    s = sd.series
    t = np.arange(sd.n, dtype=float) if sd.time_basis == "row_index" else epoch_seconds(s.index) - epoch_seconds(s.index)[0]
    if np.ptp(s.values) == 0:
        return {"direction": "flat", "slope": 0.0, "r_squared": None, "kendall_tau": None, "p_value": None, "significant": False, "provenance": sd.provenance()}
    lr = stats.linregress(t, s.values)
    tau, p = stats.kendalltau(t, s.values)
    mult, label = _rate_unit(sd)
    sig = bool(p is not None and p < alpha)
    direction = ("increasing" if tau > 0 else "decreasing") if sig else "no significant trend"
    return {
        "direction": direction,
        "slope": float(lr.slope) * mult,
        "slope_unit": f"{sd.unit + ' ' if sd.unit else ''}{label}".strip(),
        "r_squared": float(lr.rvalue**2),
        "kendall_tau": float(tau),
        "p_value": float(p),
        "significant": sig,
        "alpha": alpha,
        "provenance": sd.provenance(),
    }


def detect_peaks(sd: SeriesData, prominence: float | None = None, max_peaks: int = 20) -> dict[str, Any]:
    _require(sd, 3, "Peak detection")
    v = sd.series.values
    prom = prominence if prominence is not None else max(float(np.std(v)), 1e-12)
    out: dict[str, Any] = {}
    for kind, arr in (("peaks", v), ("troughs", -v)):
        idx, props = signal.find_peaks(arr, prominence=prom)
        order = np.argsort(props["prominences"])[::-1][:max_peaks]
        out[kind] = [
            {"timestamp": _ts(sd.series.index[i]), "value": float(v[i]), "prominence": float(props["prominences"][k])}
            for k, i in sorted(zip(order, idx[order], strict=True), key=lambda z: z[1])
        ]
    out["prominence_threshold"] = prom
    out["provenance"] = sd.provenance()
    return out


def detect_anomalies(sd: SeriesData, threshold: float = 3.5, window: int | None = None) -> dict[str, Any]:
    """Robust z-score with median/MAD (optionally rolling). Points with
    |z| ≥ threshold are anomalies. Returns nothing when variability is zero."""
    _require(sd, 5, "Anomaly detection")
    s = sd.series
    if window and window >= 5 and sd.n > window:
        med = s.rolling(window, center=True, min_periods=5).median()
        mad = (s - med).abs().rolling(window, center=True, min_periods=5).median()
    else:
        med = pd.Series(np.median(s.values), index=s.index)
        mad = pd.Series(np.median(np.abs(s.values - med.values)), index=s.index)
    scale = 1.4826 * mad
    fallback = float(s.std(ddof=1)) if sd.n > 1 else 0.0
    scale = scale.where(scale > 1e-12, fallback if fallback > 1e-12 else np.nan)
    z = (s - med) / scale
    flagged = z[z.abs() >= threshold].dropna()
    items = [
        {"timestamp": _ts(t), "value": float(s.loc[t]), "robust_z": float(z.loc[t]), "baseline_median": float(med.loc[t]), "direction": "high" if z.loc[t] > 0 else "low"}
        for t in flagged.index
    ]
    return {
        "method": "robust z-score (median/MAD" + (f", rolling {window} samples" if window else "") + ")",
        "threshold": threshold,
        "count": len(items),
        "anomalies": items[:100],
        "truncated": len(items) > 100,
        "provenance": sd.provenance(),
        "note": "" if np.isfinite(scale).any() else "The series has no variability, so no anomalies can be defined.",
    }


def threshold_violations(sd: SeriesData, upper: float | None = None, lower: float | None = None, min_duration_s: float = 0.0) -> dict[str, Any]:
    """Contiguous episodes outside [lower, upper] lasting at least ``min_duration_s``."""
    if upper is None and lower is None:
        raise ValueError("Give an upper and/or a lower threshold.")
    _require(sd, 1, "Threshold checking")
    s = sd.series
    out = pd.Series(False, index=s.index)
    if upper is not None:
        out |= s > upper
    if lower is not None:
        out |= s < lower
    episodes = []
    start = None
    prev_t = None
    for t, flag in out.items():
        if flag and start is None:
            start = t
        if not flag and start is not None:
            episodes.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None:
        episodes.append((start, s.index[-1]))
    items = []
    for a, b in episodes:
        seg = s.loc[a:b]
        dur = (b - a).total_seconds()
        if dur + 1e-9 < min_duration_s:
            continue
        worst = seg.max() if upper is not None and seg.max() > (upper or -np.inf) else seg.min()
        items.append({"start": _ts(a), "end": _ts(b), "duration_s": dur, "samples": int(seg.size), "worst_value": float(worst), "ongoing": bool(b == s.index[-1])})
    return {"upper": upper, "lower": lower, "min_duration_s": min_duration_s, "episodes": items, "count": len(items), "provenance": sd.provenance()}


def compare_windows(a: SeriesData, b: SeriesData, label_a: str = "window A", label_b: str = "window B") -> dict[str, Any]:
    """Compare two windows of (normally) the same metric. Includes a
    Mann–Whitney U test so a difference is not over-interpreted."""
    _require(a, 1, f"Comparison ({label_a})")
    _require(b, 1, f"Comparison ({label_b})")
    da, db = descriptive(a), descriptive(b)
    mean_change = percentage_change(da["mean"], db["mean"])
    test: dict[str, Any] = {"test": "Mann–Whitney U", "p_value": None, "note": ""}
    if a.n >= 3 and b.n >= 3:
        try:
            u = stats.mannwhitneyu(a.series.values, b.series.values, alternative="two-sided")
            test["p_value"] = float(u.pvalue)
        except ValueError as exc:
            test["note"] = str(exc)
    else:
        test["note"] = "Too few samples for a significance test."
    return {
        label_a: {k: da[k] for k in ("count", "mean", "median", "std", "min", "max", "first_at", "last_at")},
        label_b: {k: db[k] for k in ("count", "mean", "median", "std", "min", "max", "first_at", "last_at")},
        "mean_difference": db["mean"] - da["mean"],
        "mean_change_pct": mean_change["value"],
        "mean_change_note": mean_change["reason"],
        "max_difference": db["max"] - da["max"],
        "significance": test,
        "direction": "higher" if db["mean"] > da["mean"] else "lower" if db["mean"] < da["mean"] else "unchanged",
        "provenance": {label_a: a.provenance(), label_b: b.provenance()},
    }


def align_series(a: SeriesData, b: SeriesData, tolerance_s: float | None = None) -> pd.DataFrame:
    """Pair samples of two series by nearest timestamp within ``tolerance_s``
    (default: 1.5 × the coarser median sampling interval)."""
    def med_dt(sd: SeriesData) -> float:
        if sd.n < 2:
            return 1.0
        return float(np.median(np.diff(epoch_seconds(sd.series.index))))

    tol = tolerance_s if tolerance_s is not None else 1.5 * max(med_dt(a), med_dt(b))
    left = a.series.rename("a").reset_index()
    right = b.series.rename("b").reset_index()
    left.columns = ["timestamp", "a"]
    right.columns = ["timestamp", "b"]
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True).dt.as_unit("ns")
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True).dt.as_unit("ns")
    merged = pd.merge_asof(left.sort_values("timestamp"), right.sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=max(tol, 1e-3)))
    return merged.dropna()


def correlation(a: SeriesData, b: SeriesData, tolerance_s: float | None = None) -> dict[str, Any]:
    pairs = align_series(a, b, tolerance_s)
    n = len(pairs)
    res: dict[str, Any] = {
        "metric_a": a.metric,
        "metric_b": b.metric,
        "paired_samples": n,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "caution": "Correlation shows association only; it does not establish that one metric causes the other.",
        "provenance": {"a": a.provenance(), "b": b.provenance()},
    }
    if n < 3:
        res["note"] = "Fewer than 3 time-aligned sample pairs."
        return res
    if np.ptp(pairs["a"]) == 0 or np.ptp(pairs["b"]) == 0:
        res["note"] = "One series is constant in the overlapping period, so correlation is undefined."
        return res
    pr = stats.pearsonr(pairs["a"], pairs["b"])
    sr = stats.spearmanr(pairs["a"], pairs["b"])
    res.update(
        pearson_r=float(pr.statistic),
        pearson_p=float(pr.pvalue),
        spearman_rho=float(sr.statistic),
        spearman_p=float(sr.pvalue),
    )
    return res


def lagged_correlation(a: SeriesData, b: SeriesData, max_lag_steps: int = 10) -> dict[str, Any]:
    """Which lag (b shifted relative to a, in resampled steps) maximizes |r|.
    A lead of one series is still not proof of causation."""
    pairs = align_series(a, b)
    if len(pairs) < max(8, 2 * max_lag_steps):
        return {"best_lag_steps": None, "note": "Not enough aligned samples for a lag analysis."}
    best = (0, 0.0)
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        x = pairs["a"].values
        y = np.roll(pairs["b"].values, -lag)
        sl = slice(max(0, -lag), len(x) - max(0, lag))
        if np.ptp(x[sl]) == 0 or np.ptp(y[sl]) == 0:
            continue
        r = float(np.corrcoef(x[sl], y[sl])[0, 1])
        if abs(r) > abs(best[1]):
            best = (lag, r)
    step = float(np.median(np.diff(epoch_seconds(pairs["timestamp"]))))
    return {"best_lag_steps": best[0], "best_lag_seconds": best[0] * step, "r_at_best_lag": best[1], "step_seconds": step, "caution": "A lead/lag pattern is still only an association."}


def rank_rates(series: list[SeriesData]) -> dict[str, Any]:
    """Rank metrics by how fast they change. Absolute rates of different units
    are not comparable, so ranking uses the rate relative to each metric's mean level."""
    rows, skipped = [], []
    for sd in series:
        try:
            r = rate_of_change(sd)
        except (InsufficientData, ValueError) as exc:
            skipped.append({"metric": sd.metric, "source": sd.source, "reason": str(exc)})
            continue
        rows.append(
            {
                "metric": sd.metric,
                "source": sd.source,
                "status": sd.status,
                "unit": sd.unit,
                "slope": r["slope"],
                "rate_unit": r["rate_unit"],
                "relative_rate_pct": r["relative_rate_pct"],
                "n": sd.n,
                "from": sd.series.index[0].isoformat(),
                "to": sd.series.index[-1].isoformat(),
            }
        )
    rows.sort(key=lambda r: abs(r["relative_rate_pct"]) if r["relative_rate_pct"] is not None else -1, reverse=True)
    return {
        "ranking": rows,
        "skipped": skipped,
        "method": "least-squares slope over actual timestamps, ranked by |slope| as % of each metric's mean level per minute",
    }


def data_quality(sd: SeriesData, expected_interval_s: float | None = None) -> dict[str, Any]:
    """0..1 score combining completeness against the expected sampling interval
    and the cleaning notes."""
    if sd.n == 0:
        return {"score": 0.0, "completeness": 0.0, "notes": ["No data."]}
    if sd.n == 1 or sd.time_basis == "row_index":
        return {"score": 0.85 if sd.n > 1 else 0.5, "completeness": None, "notes": sd.notes}
    dt = np.diff(epoch_seconds(sd.series.index))
    expected = expected_interval_s or float(np.median(dt))
    span = (sd.series.index[-1] - sd.series.index[0]).total_seconds()
    expected_n = span / expected + 1 if expected > 0 else sd.n
    completeness = float(min(1.0, sd.n / expected_n)) if expected_n > 0 else 1.0
    penalty = 0.05 * len([n for n in sd.notes if "duplicate" in n or "excluded" in n])
    est_penalty = 0.15 if sd.status == "estimated" else 0.0
    return {
        "score": round(max(0.0, completeness - penalty - est_penalty), 3),
        "completeness": round(completeness, 3),
        "expected_interval_s": expected,
        "largest_gap_s": float(dt.max()),
        "notes": sd.notes,
    }
