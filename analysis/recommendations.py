"""Recommendations with measurable success criteria and before/after evaluation.

An evaluation reports whether the *criterion* was met after the
recommendation. It never claims the recommendation caused the change.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from core.models import utcnow
from storage.db import Database
from analysis.stats import InsufficientData, compare_windows, prepare_series

VALID_DIRECTIONS = ("decrease", "increase", "below", "above", "within")


def make_criteria(metric: str, direction: str, target: float | None = None, lower: float | None = None, window_s: float = 600.0) -> dict[str, Any]:
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {VALID_DIRECTIONS}")
    if direction in ("below", "above") and target is None:
        raise ValueError(f"'{direction}' criteria need a target value.")
    if direction == "within" and (target is None or lower is None):
        raise ValueError("'within' criteria need lower and upper bounds.")
    if window_s <= 0:
        raise ValueError("The evaluation window must be positive.")
    return {"metric": metric, "direction": direction, "target": target, "lower": lower, "window_s": float(window_s)}


def describe_criteria(c: dict[str, Any]) -> str:
    m, d = c["metric"], c["direction"]
    w = f"over the {int(c['window_s'] // 60) or c['window_s']} {'min' if c['window_s'] >= 60 else 's'} after the recommendation"
    if d == "below":
        return f"{m} mean stays below {c['target']:g} {w}"
    if d == "above":
        return f"{m} mean stays above {c['target']:g} {w}"
    if d == "within":
        return f"{m} stays within [{c['lower']:g}, {c['target']:g}] {w}"
    return f"{m} mean {d}s compared with the same-length window before {w.replace('over the', 'measured over the')}"


def evaluate(db: Database, rec: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    now = now or utcnow()
    c = rec["criteria"]
    created = datetime.fromisoformat(rec["created_at"])
    w = timedelta(seconds=c["window_s"])
    before = db.query_observations(metric=c["metric"], start=created - w, end=created, source_id=rec.get("source_id") or None)
    after_end = min(now, created + w)
    after = db.query_observations(metric=c["metric"], start=created, end=after_end, source_id=rec.get("source_id") or None)
    result: dict[str, Any] = {
        "recommendation_id": rec["id"],
        "criterion": describe_criteria(c),
        "before_window": {"start": (created - w).isoformat(), "end": created.isoformat(), "n": len(before)},
        "after_window": {"start": created.isoformat(), "end": after_end.isoformat(), "n": len(after)},
        "complete": now >= created + w,
        "caution": "This compares measurements before and after the recommendation. It shows whether the criterion was met, not that the recommendation caused the change.",
    }
    if after.empty:
        result.update(status="insufficient_data", met=None, reason="No measurements after the recommendation yet.")
        return result
    sa = prepare_series(after, c["metric"])
    mean_after = float(sa.series.mean())
    result["after_mean"] = mean_after
    if c["direction"] in ("decrease", "increase"):
        if before.empty:
            result.update(status="insufficient_data", met=None, reason="No measurements before the recommendation to compare against.")
            return result
        sb = prepare_series(before, c["metric"])
        try:
            cmp = compare_windows(sb, sa, "before", "after")
        except InsufficientData as exc:
            result.update(status="insufficient_data", met=None, reason=str(exc))
            return result
        result["comparison"] = cmp
        result["before_mean"] = cmp["before"]["mean"]
        met = cmp["mean_difference"] < 0 if c["direction"] == "decrease" else cmp["mean_difference"] > 0
        p = cmp["significance"]["p_value"]
        result["statistically_clear"] = bool(p is not None and p < 0.05)
    elif c["direction"] == "below":
        met = bool((sa.series < c["target"]).all())
        result["share_below_target"] = float((sa.series < c["target"]).mean())
    elif c["direction"] == "above":
        met = bool((sa.series > c["target"]).all())
        result["share_above_target"] = float((sa.series > c["target"]).mean())
    else:
        inside = (sa.series >= c["lower"]) & (sa.series <= c["target"])
        met = bool(inside.all())
        result["share_within"] = float(inside.mean())
    result["met"] = bool(met)
    result["status"] = ("met" if met else "not_met") if result["complete"] else ("on_track" if met else "not_met_so_far")
    return result


def suggest_criteria(metric: str, stats_desc: dict[str, Any], anomalies: dict[str, Any] | None, alert: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive a measurable default criterion from the evidence the advice was based on."""
    if alert and alert.get("evidence", {}).get("latest") is not None and "<" not in alert.get("message", "") and ">" in alert.get("message", ""):
        try:
            thr = float(alert["message"].split(">")[1].split()[0])
            return make_criteria(metric, "below", thr)
        except (ValueError, IndexError):
            pass
    if anomalies and anomalies.get("anomalies"):
        last = anomalies["anomalies"][-1]
        return make_criteria(metric, "decrease" if last["direction"] == "high" else "increase")
    if stats_desc.get("change") is not None and stats_desc["change"] > 0:
        return make_criteria(metric, "decrease")
    return make_criteria(metric, "increase")


def obs_frame_to_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df["value"].values, index=pd.to_datetime(df["timestamp"], utc=True))
