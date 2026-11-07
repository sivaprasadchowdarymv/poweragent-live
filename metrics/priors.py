"""Benchmark-informed confidence: scale each reader's live readings by the
accuracy it actually achieved on ChartNet for that chart type."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def latest_run(data_dir: Path) -> Path | None:
    root = Path(data_dir) / "benchmarks"
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if (p / "summary.json").exists()), key=lambda p: p.name)
    return runs[-1] if runs else None


def load_priors(data_dir: Path) -> dict[str, Any]:
    run = latest_run(data_dir)
    if run is None or not (run / "priors.json").exists():
        return {}
    try:
        data = json.loads((run / "priors.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    pri = data.get("priors") or {}
    pri["_run"] = run.name
    return pri


def prior_factor(priors: dict[str, Any], reader: str, chart_type: str) -> float | None:
    """Measured F1@5% for this reader and chart type (falls back to the reader's
    overall score). ``None`` when no benchmark has been run."""
    table = priors.get(reader)
    if not table:
        return None
    ct = (chart_type or "").lower()
    for key, val in table.items():
        if key.startswith("_"):
            continue
        k = key.lower()
        if ct and (k == ct or ct in k.split() or k.startswith(ct) or ct.startswith(k)):
            return float(val)
    overall = table.get("_overall")
    return float(overall) if overall is not None else None


def scaled(confidence: float, factor: float | None) -> float:
    """Confidence × measured accuracy, but never below 0.05 or above the original."""
    if factor is None:
        return confidence
    return round(max(0.05, min(confidence, confidence * factor)), 3)
