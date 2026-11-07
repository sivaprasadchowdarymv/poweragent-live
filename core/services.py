"""DataService: the single place that knows where measurements live.

It owns uploaded datasets, the latest screen analysis, stored observations,
source preferences, generated figures and reports. Tools and UI call this
service; nothing else touches the database directly.
"""

from __future__ import annotations

import difflib
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any

import pandas as pd

from ingest.files import Dataset, normalize_dataset
from analysis.stats import SeriesData, prepare_series
from analysis.windows import parse_window
from metrics.priors import prior_factor, scaled
from core.models import SourceType, TimeWindow, ValueStatus, new_id, utcnow
from storage.db import Database, metric_key
from vision.chart_reader import ScreenAnalysis, time_axis_to_datetimes

log = logging.getLogger(__name__)
SCREEN_SOURCE_ID = "screen_live"
CHART_MODEL_SOURCE_ID = "screen_chart_model"


class DataService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.RLock()
        self.datasets: dict[str, Dataset] = {}
        self.latest_screen: ScreenAnalysis | None = None
        self.screen_history: deque[dict[str, Any]] = deque(maxlen=200)  # bounded summaries, no pixels
        self.vision_notes: dict[str, Any] | None = None
        self.vision_history: deque[dict[str, Any]] = deque(maxlen=60)
        self.chart_model: dict[str, Any] | None = None  # latest Chart-model readings (summaries, no pixels)
        self.preferred_source: dict[str, str] = {}  # metric_key -> source_id chosen by the user
        self.priors: dict[str, Any] = {}  # ChartNet benchmark accuracy per reader/chart type
        self.figures: dict[str, dict[str, Any]] = {}
        self.reports: dict[str, str] = {}
        self._figure_order: deque[str] = deque(maxlen=100)

    # -------------------------------------------------------------- files
    def ingest_dataset(self, ds: Dataset) -> dict[str, Any]:
        obs, report = normalize_dataset(ds)
        with self._lock:
            # Re-ingesting the same dataset replaces its observations, never duplicates them.
            self.db.delete_source(ds.id)
            self.db.upsert_source(
                ds.id,
                SourceType.FILE.value,
                ds.name,
                {"sha256": ds.sha256, "size_bytes": ds.size_bytes, "timestamp_col": ds.timestamp_col, "metrics": ds.metric_cols, "time_basis": ds.time_basis, "quality": report.get("quality_score")},
            )
            n = self.db.add_observations(obs)
            self.datasets[ds.id] = ds
        report["stored"] = n
        return report

    def remove_dataset(self, ds_id: str) -> None:
        with self._lock:
            self.datasets.pop(ds_id, None)
            self.db.delete_source(ds_id)

    # -------------------------------------------------------------- screen
    def add_screen_analysis(self, analysis: ScreenAnalysis) -> int:
        rows = screen_observations(analysis, self.priors)
        with self._lock:
            self.latest_screen = analysis
            self.screen_history.append(analysis.summary())
            if rows:
                self.db.upsert_source(SCREEN_SOURCE_ID, SourceType.SCREEN.value, "Live screen", {"target": "screen"})
                return self.db.add_observations(pd.DataFrame(rows))
        return 0

    def set_vision_notes(self, notes: dict[str, Any], frame_seq: int, captured_at: datetime) -> int:
        """Store a vision-model reading: latest + history + SQLite, and its values as observations."""
        from vision.inference import VISION_SOURCE_ID, observations

        with self._lock:
            self.vision_notes = {"frame_seq": frame_seq, "captured_at": captured_at.isoformat(), **notes}
            self.vision_history.append(self.vision_notes)
            ocr = "\n".join(self.latest_screen.text_lines) if self.latest_screen is not None and self.latest_screen.frame_seq == frame_seq else ""
        try:
            self.db.save_analysis("vision_reading", {"frame_seq": frame_seq}, self.vision_notes)
        except Exception:  # noqa: BLE001
            pass
        rows = observations(notes, captured_at, frame_seq, ocr)
        if rows:
            self.db.upsert_source(VISION_SOURCE_ID, SourceType.SCREEN.value, "Live screen (vision model)", {"model": notes.get("model", "")})
            return self.db.add_observations(pd.DataFrame(rows))
        return 0

    def add_chart_model_readings(self, readings: list[Any], frame_seq: int, captured_at: datetime) -> int:
        """Store chart-model chart-to-table readings as their own estimated source.
        Per chart and series, the last table row becomes an observation at the
        capture time (or at the chart's own clock time when the x column is a time)."""
        from vision.chart_model import fused_confidence, series_metric_name, x_value
        from vision.chart_reader import time_axis_to_datetimes

        rows: list[dict[str, Any]] = []
        for r in readings:
            if not r.table.ok:
                continue
            from metrics.priors import prior_factor, scaled

            chart_type = next((c.chart_type for c in (self.latest_screen.charts if self.latest_screen else []) if c.id == r.chart_id), "")
            factor = prior_factor(self.priors, "model", chart_type)
            conf = scaled(fused_confidence(r.check), factor)
            for sname, pts in r.table.series.items():
                if not pts:
                    continue
                label, value = pts[-1]
                ts = captured_at
                if r.table.x_kind == "time":
                    xv = x_value(label, "time")
                    if xv is not None:
                        ts = time_axis_to_datetimes([xv], captured_at)[0]
                rows.append(
                    {
                        "timestamp": ts,
                        "source_type": SourceType.SCREEN.value,
                        "source_id": CHART_MODEL_SOURCE_ID,
                        "metric": series_metric_name(sname, r.chart_title or r.table.title),
                        "value": float(value),
                        "unit": "",
                        "confidence": conf,
                        "status": ValueStatus.ESTIMATED.value,
                        "meta": {"frame_seq": frame_seq, "chart_id": r.chart_id, "method": f"{r.model} chart-to-table", "x_label": label, "cross_check": r.check.status, "series_name": sname, "benchmark_f1": factor},
                    }
                )
        with self._lock:
            self.chart_model = {
                "frame_seq": frame_seq,
                "captured_at": captured_at.isoformat(),
                "readings": [r.summary() for r in readings],
            }
            if rows:
                self.db.upsert_source(CHART_MODEL_SOURCE_ID, SourceType.SCREEN.value, "Live screen (chart model)", {"model": readings[0].model if readings else ""})
                # Use the pixel reader's unit for the same chart when available.
                if self.latest_screen is not None:
                    units = {c.id: c.unit for c in self.latest_screen.charts}
                    for row in rows:
                        row["unit"] = units.get(row["meta"]["chart_id"], "")
                return self.db.add_observations(pd.DataFrame(rows))
        return 0

    def screen_context(self) -> dict[str, Any]:
        with self._lock:
            a = self.latest_screen
            if a is None:
                return {"available": False, "reason": "No screen has been captured and analyzed yet."}
            age = (utcnow() - a.captured_at).total_seconds()
            out = a.summary()
            out["available"] = True
            out["age_seconds"] = round(age, 1)
            out["vision_model_notes"] = self.vision_notes
            if self.chart_model:
                out["chart_model_readings"] = self.chart_model
                if self.chart_model.get("frame_seq") != a.frame_seq:
                    out["chart_model_note"] = "Chart-model readings are from an earlier frame than the latest pixel/OCR analysis."
            if self.vision_notes and self.vision_notes.get("frame_seq") != a.frame_seq:
                out["vision_model_notes_note"] = "Vision-model notes are from an earlier frame than the latest OCR analysis."
            return out

    # -------------------------------------------------------------- catalog & queries
    def metric_catalog(self) -> list[dict[str, Any]]:
        return self.db.metric_catalog()

    def metric_names(self) -> list[str]:
        seen: dict[str, str] = {}
        for row in self.metric_catalog():
            seen.setdefault(row["metric_key"], row["metric"])
        return list(seen.values())

    def resolve_metric(self, name: str | None) -> str | None:
        """Match a user/LLM metric reference to a known metric (case/units/fuzzy)."""
        if not name:
            return None
        names = self.metric_names()
        key = metric_key(name)
        by_key = {metric_key(n): n for n in names}
        if key in by_key:
            return by_key[key]
        contains = [n for k, n in by_key.items() if key and (key in k or k in key)]
        if len(contains) == 1:
            return contains[0]
        close = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.72)
        return by_key[close[0]] if close else None

    def sources_for(self, metric: str) -> list[dict[str, Any]]:
        k = metric_key(metric)
        return [r for r in self.metric_catalog() if r["metric_key"] == k]

    def choose_source(self, metric: str, source_id: str | None = None) -> tuple[str | None, list[dict[str, Any]], str]:
        """Pick a source for a metric. Order: explicit > user preference >
        verified file data > corrected > estimated screen data. Returns the
        chosen id, all candidates, and a one-line reason."""
        cands = self.sources_for(metric)
        if not cands:
            return None, [], "No data for this metric."
        ids = {c["source_id"] for c in cands}
        if source_id:
            if source_id in ids:
                return source_id, cands, "Source requested explicitly."
            match = [c["source_id"] for c in cands if source_id.lower() in (c["source_id"] + c["source_type"]).lower()]
            if match:
                return match[0], cands, "Source requested explicitly."
        pref = self.preferred_source.get(metric_key(metric))
        if pref in ids:
            return pref, cands, "User-selected authoritative source."
        rank = {ValueStatus.VERIFIED.value: 0, ValueStatus.CORRECTED.value: 1, ValueStatus.ESTIMATED.value: 2}
        # Among equally ranked sources (e.g. two screen readers), prefer the more confident, then the most recent.
        best = sorted(cands, key=lambda c: (rank.get(c["status"], 3), -round(c["mean_confidence"], 2), -c["last"].timestamp()))[0]
        reason = "Verified structured data preferred over screen estimates." if len(ids) > 1 else "Only source for this metric."
        return best["source_id"], cands, reason

    def get_series(self, metric: str, window: str | TimeWindow | None = None, source_id: str | None = None) -> tuple[SeriesData, TimeWindow, dict[str, Any]]:
        resolved = self.resolve_metric(metric)
        if resolved is None:
            known = ", ".join(self.metric_names()[:20]) or "none yet"
            raise KeyError(f"No data for metric '{metric}'. Known metrics: {known}.")
        sid, cands, reason = self.choose_source(resolved, source_id)
        info = [c for c in cands if c["source_id"] == sid]
        data_end = max(c["last"] for c in info) if info else None
        data_start = min(c["first"] for c in info) if info else None
        win = window if isinstance(window, TimeWindow) else parse_window(window, data_end=data_end, data_start=data_start)
        df = self.db.query_observations(metric=resolved, start=win.start, end=win.end, source_id=sid)
        statuses = set(df["status"]) if not df.empty else {info[0]["status"] if info else "verified"}
        status = statuses.pop() if len(statuses) == 1 else "mixed"
        unit = info[0]["unit"] if info else ""
        time_basis = "row_index" if (not df.empty and df["meta"].map(lambda m: m.get("time_basis") == "row_index").any()) else "timestamp"
        sd = prepare_series(df, resolved, unit, sid or "", status, time_basis) if not df.empty else SeriesData(pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")), resolved, unit, sid or "", status, time_basis)
        if win.anchor == "latest-data":
            sd.notes.append(f"Window is relative to the latest data timestamp ({data_end.isoformat() if data_end else '?'}), not the current time; this data is historical.")
        meta = {
            "metric": resolved,
            "source_id": sid,
            "source_reason": reason,
            "other_sources": [c["source_id"] for c in cands if c["source_id"] != sid],
            "window": win.to_dict(),
            "window_description": win.describe(),
            "data_age_seconds": (utcnow() - data_end).total_seconds() if data_end else None,
        }
        return sd, win, meta

    # -------------------------------------------------------------- figures & reports
    def add_figure(self, fig_json: dict[str, Any], description: dict[str, Any]) -> str:
        fid = new_id("fig")
        with self._lock:
            self.figures[fid] = {"figure": fig_json, "description": description, "created_at": utcnow().isoformat()}
            self._figure_order.append(fid)
            while len(self.figures) > 100:
                self.figures.pop(self._figure_order.popleft(), None)
        return fid

    def add_report(self, markdown: str) -> str:
        rid = new_id("rep")
        with self._lock:
            self.reports[rid] = markdown
            if len(self.reports) > 20:
                self.reports.pop(next(iter(self.reports)))
        return rid


def screen_observations(a: ScreenAnalysis, priors: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Each analyzed frame contributes the latest value of each detected series
    and each KPI tile, stamped with the capture time and marked as estimates.
    When the chart has a clock-time x-axis the latest point uses the chart's own
    time instead of the capture time."""
    rows: list[dict[str, Any]] = []
    for chart in a.charts:
        for s in chart.series:
            if s.latest is None:
                continue
            ts = a.captured_at
            if chart.x_axis is not None and chart.x_axis.scale == "time" and s.x:
                ts = time_axis_to_datetimes([max(s.x)], a.captured_at)[0]
            name = s.name if not s.name.startswith("series@") else f"{chart.title or chart.id} {s.name}"
            rows.append(
                {
                    "timestamp": ts,
                    "source_type": SourceType.SCREEN.value,
                    "source_id": SCREEN_SOURCE_ID,
                    "metric": _strip_unit(name),
                    "value": float(s.latest),
                    "unit": chart.unit,
                    "confidence": scaled(s.confidence, factor := prior_factor(priors or {}, "pixel", chart.chart_type)),
                    "status": ValueStatus.ESTIMATED.value,
                    "meta": {"frame_seq": a.frame_seq, "chart_id": chart.id, "chart_title": chart.title, "kind": s.kind, "method": "pixel-to-axis calibration", "clipped": s.clipped, "benchmark_f1": factor},
                }
            )
    for k in a.kpis:
        rows.append(
            {
                "timestamp": a.captured_at,
                "source_type": SourceType.SCREEN.value,
                "source_id": SCREEN_SOURCE_ID,
                "metric": _strip_unit(k.label),
                "value": float(k.value),
                "unit": k.unit,
                "confidence": k.confidence,
                "status": ValueStatus.ESTIMATED.value,
                "meta": {"frame_seq": a.frame_seq, "method": "OCR of KPI text", "raw_text": k.raw_text[:80]},
            }
        )
    return rows


def _strip_unit(name: str) -> str:
    import re

    return re.sub(r"\s*[\(\[][^\)\]]{1,12}[\)\]]\s*$", "", name).strip() or name
