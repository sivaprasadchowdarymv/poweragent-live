"""Compare the same metric across sources (e.g. screen estimate vs. file).

Sources are never merged silently: this module only *reports* agreement and
discrepancies; the user chooses which source is authoritative.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.stats import SeriesData, align_series


def compare_sources(
    a: SeriesData,
    b: SeriesData,
    tolerance_s: float | None = None,
    abs_tol: float | None = None,
    rel_tol: float = 0.02,
) -> dict[str, Any]:
    pairs = align_series(a, b, tolerance_s)
    base: dict[str, Any] = {
        "metric": a.metric,
        "source_a": {"id": a.source, "status": a.status, "n": a.n, "unit": a.unit},
        "source_b": {"id": b.source, "status": b.status, "n": b.n, "unit": b.unit},
        "unit_mismatch": bool(a.unit and b.unit and a.unit.lower() != b.unit.lower()),
        "paired_samples": len(pairs),
        "policy": "Sources are not merged. Choose an authoritative source in Data Sources if they disagree.",
    }
    if base["unit_mismatch"]:
        base["warning"] = f"Units differ ({a.unit} vs {b.unit}); values are compared as-is and may not be comparable."
    if pairs.empty:
        base["note"] = "The two sources have no samples close enough in time to compare."
        overlap = _overlap(a, b)
        base["time_overlap"] = overlap
        return base
    diff = pairs["b"] - pairs["a"]
    scale = np.maximum(np.abs(pairs["a"].values), 1e-12)
    limit = np.maximum(abs_tol if abs_tol is not None else 0.0, rel_tol * scale)
    bad = np.abs(diff.values) > limit
    disc = pairs[bad].copy()
    disc["difference"] = diff[bad]
    base.update(
        mean_difference=float(diff.mean()),
        mean_abs_difference=float(diff.abs().mean()),
        max_abs_difference=float(diff.abs().max()),
        mean_relative_difference_pct=float(np.mean(np.abs(diff.values) / scale) * 100),
        tolerance={"relative": rel_tol, "absolute": abs_tol},
        discrepancies=int(bad.sum()),
        agreement_pct=float(100 * (1 - bad.mean())),
        examples=[
            {"timestamp": r.timestamp.isoformat(), "a": float(r.a), "b": float(r.b), "difference": float(r.difference)}
            for r in disc.head(10).itertuples()
        ],
    )
    return base


def _overlap(a: SeriesData, b: SeriesData) -> dict[str, Any] | None:
    if a.n == 0 or b.n == 0:
        return None
    start = max(a.series.index[0], b.series.index[0])
    end = min(a.series.index[-1], b.series.index[-1])
    return {"start": start.isoformat(), "end": end.isoformat(), "overlaps": bool(end >= start)}
