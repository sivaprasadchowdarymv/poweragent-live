"""Controlled tool registry.

The language model can only call the functions registered here. Every
argument is validated (type, enum, range, length) before execution, and every
result is a bounded JSON-serializable dict. There is deliberately no shell,
no code execution and no filesystem access.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from analysis import plots
from analysis import recommendations as recs
from analysis.alerts import AlertEngine
from analysis.reconcile import compare_sources
from analysis.stats import (
    InsufficientData,
    compare_windows,
    correlation,
    data_quality,
    descriptive,
    detect_anomalies,
    detect_peaks,
    lagged_correlation,
    moving_average,
    rank_rates,
    rate_of_change,
    trend,
)
from analysis.windows import WindowError, parse_window
from core.models import TimeWindow, utcnow
from core.services import DataService

log = logging.getLogger(__name__)
MAX_RESULT_CHARS = 12_000


class ToolError(ValueError):
    """Invalid tool call; message is returned to the model as the tool result."""


@dataclass
class Param:
    name: str
    type: str  # string | number | integer | boolean | array
    description: str
    required: bool = False
    enum: list[str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    max_length: int = 200
    max_items: int = 8

    def schema(self) -> dict[str, Any]:
        s: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            s["enum"] = self.enum
        if self.type == "array":
            s["items"] = {"type": "string"}
        return s

    def validate(self, value: Any) -> Any:
        if self.type == "string":
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if len(value) > self.max_length:
                raise ToolError(f"'{self.name}' is too long (max {self.max_length} characters).")
            if self.enum and value not in self.enum:
                raise ToolError(f"'{self.name}' must be one of {self.enum}.")
            return value
        if self.type in ("number", "integer"):
            if isinstance(value, bool):
                raise ToolError(f"'{self.name}' must be a number.")
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise ToolError(f"'{self.name}' must be a number.") from exc
            if not np.isfinite(num):
                raise ToolError(f"'{self.name}' must be finite.")
            if self.minimum is not None and num < self.minimum or self.maximum is not None and num > self.maximum:
                raise ToolError(f"'{self.name}' must be between {self.minimum} and {self.maximum}.")
            return int(num) if self.type == "integer" else num
        if self.type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes")
            return bool(value)
        if self.type == "array":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = [v.strip() for v in value.split(",") if v.strip()]
            if not isinstance(value, list):
                raise ToolError(f"'{self.name}' must be a list.")
            if len(value) > self.max_items:
                raise ToolError(f"'{self.name}' accepts at most {self.max_items} items.")
            return [str(v).strip()[: self.max_length] for v in value]
        raise ToolError(f"Unsupported parameter type {self.type}.")  # pragma: no cover


@dataclass
class Tool:
    name: str
    description: str
    params: list[Param]
    fn: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p.name: p.schema() for p in self.params},
                    "required": [p.name for p in self.params if p.required],
                },
            },
        }


@dataclass
class ToolContext:
    service: DataService
    alerts: AlertEngine
    conversation_id: str | None = None
    focus: dict[str, Any] = field(default_factory=dict)  # updated by tools for follow-up resolution
    chart_model: Any = None  # ChartModelClient when a chart model is enabled
    latest_image: Any = None  # callable returning the latest captured PIL image


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    ms: float


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def validate(self, name: str, arguments: Any) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(self.tools)}.")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError as exc:
                raise ToolError("Tool arguments must be a JSON object.") from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object.")
        known = {p.name: p for p in tool.params}
        unknown = set(arguments) - set(known)
        if unknown:
            raise ToolError(f"Unknown argument(s) for {name}: {', '.join(sorted(unknown))}.")
        clean: dict[str, Any] = {}
        for p in tool.params:
            if p.name in arguments and arguments[p.name] not in (None, ""):
                clean[p.name] = p.validate(arguments[p.name])
            elif p.required:
                raise ToolError(f"Missing required argument '{p.name}' for {name}.")
        return clean

    def call(self, ctx: ToolContext, name: str, arguments: Any) -> ToolCallRecord:
        import time

        t0 = time.perf_counter()
        try:
            args = self.validate(name, arguments)
            result = self.tools[name].fn(ctx, **args)
            ok = True
        except (ToolError, KeyError, InsufficientData, WindowError, ValueError) as exc:
            args = arguments if isinstance(arguments, dict) else {}
            result = {"error": str(exc).strip("'\"")}
            ok = False
        except Exception as exc:  # never let a tool crash the chat
            log.exception("Tool %s failed", name)
            args = arguments if isinstance(arguments, dict) else {}
            result = {"error": f"Internal error in {name}: {exc}"}
            ok = False
        return ToolCallRecord(name, args, bound(result), ok, (time.perf_counter() - t0) * 1000)


def bound(result: dict[str, Any]) -> dict[str, Any]:
    """Keep tool results small enough for the model context."""
    text = json.dumps(result, default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return json.loads(text)
    trimmed = dict(result)
    for key in ("points", "anomalies", "examples", "episodes", "peaks", "troughs", "ranking", "history"):
        if key in trimmed and isinstance(trimmed[key], list):
            trimmed[key] = trimmed[key][:15]
            trimmed[f"{key}_truncated"] = True
    text = json.dumps(trimmed, default=str)
    if len(text) > MAX_RESULT_CHARS:
        return {"truncated": True, "summary": text[:MAX_RESULT_CHARS]}
    return json.loads(text)


# =================================================================== tool functions
def _series(ctx: ToolContext, metric: str, window: str | None, source: str | None = None) -> tuple[Any, TimeWindow, dict[str, Any]]:
    sd, win, meta = ctx.service.get_series(metric, window or "all", source)
    ctx.focus.update(metric=sd.metric, source_id=meta["source_id"], window=win.to_dict())
    return sd, win, meta


def t_screen(ctx: ToolContext) -> dict[str, Any]:
    out = ctx.service.screen_context()
    if out.get("available") and out.get("charts"):
        ctx.focus["chart_id"] = out["charts"][0]["id"]
        ctx.focus["charts"] = [c["id"] for c in out["charts"]]
    return out


def t_list_charts(ctx: ToolContext) -> dict[str, Any]:
    sc = ctx.service.screen_context()
    if not sc.get("available"):
        return {"charts": [], "note": sc.get("reason")}
    return {"captured_at": sc["captured_at"], "age_seconds": sc["age_seconds"], "charts": sc["charts"], "kpis": sc["kpis"]}


def t_list_metrics(ctx: ToolContext) -> dict[str, Any]:
    rows = ctx.service.metric_catalog()
    now = utcnow()
    return {
        "metrics": [
            {
                "metric": r["metric"],
                "source_id": r["source_id"],
                "source_type": r["source_type"],
                "status": r["status"],
                "unit": r["unit"],
                "count": r["count"],
                "first": r["first"].isoformat(),
                "last": r["last"].isoformat(),
                "last_age_seconds": round((now - r["last"]).total_seconds(), 1),
                "mean_confidence": r["mean_confidence"],
            }
            for r in rows[:60]
        ],
        "preferred_sources": ctx.service.preferred_source,
    }


def t_history(ctx: ToolContext, metric: str, window: str | None = None, source: str | None = None, max_points: int = 30) -> dict[str, Any]:
    sd, win, meta = _series(ctx, metric, window or "last 10 minutes", source)
    s = sd.series
    if s.size > max_points:
        idx = np.linspace(0, s.size - 1, max_points).round().astype(int)
        s = s.iloc[idx]
    return {**meta, "status": sd.status, "unit": sd.unit, "n_total": sd.n, "points": [{"timestamp": t.isoformat(), "value": float(v)} for t, v in s.items()], "notes": sd.notes}


def t_stats(ctx: ToolContext, metric: str, window: str | None = None, source: str | None = None) -> dict[str, Any]:
    sd, win, meta = _series(ctx, metric, window, source)
    out: dict[str, Any] = {**meta, "statistics": descriptive(sd), "quality": data_quality(sd)}
    if sd.n >= 2:
        try:
            out["rate_of_change"] = rate_of_change(sd)
        except InsufficientData as exc:
            out["rate_of_change"] = {"note": str(exc)}
    if sd.n >= 3:
        out["trend"] = trend(sd)
    ctx.service.db.save_analysis("statistics", {"metric": sd.metric, "window": win.to_dict(), "source": meta["source_id"]}, out)
    return out


def t_compare_windows(ctx: ToolContext, metric: str, window_a: str = "previous 10 minutes", window_b: str = "last 10 minutes", source: str | None = None) -> dict[str, Any]:
    resolved = ctx.service.resolve_metric(metric) or metric
    sid, cands, _ = ctx.service.choose_source(resolved, source)
    info = [c for c in cands if c["source_id"] == sid]
    end = max(c["last"] for c in info) if info else None
    now = utcnow()  # one clock reading so adjacent windows share an exact boundary
    wa = parse_window(window_a, data_end=end, now=now)
    wb = parse_window(window_b, data_end=end, now=now)
    a, _, meta_a = ctx.service.get_series(resolved, wa, sid)
    b, _, _ = ctx.service.get_series(resolved, wb, sid)
    res = compare_windows(a, b, wa.label or "window A", wb.label or "window B")
    res["windows"] = {"a": wa.to_dict() | {"description": wa.describe()}, "b": wb.to_dict() | {"description": wb.describe()}}
    res["source_id"] = sid
    ctx.focus.update(metric=a.metric, source_id=sid, window=wb.to_dict())
    ctx.service.db.save_analysis("compare_windows", {"metric": a.metric, "a": wa.to_dict(), "b": wb.to_dict()}, res)
    return res


def t_rank_rates(ctx: ToolContext, window: str | None = None) -> dict[str, Any]:
    series, errors = [], []
    seen = set()
    for row in ctx.service.metric_catalog():
        if row["metric_key"] in seen:
            continue
        seen.add(row["metric_key"])
        try:
            sd, win, meta = ctx.service.get_series(row["metric"], window or "last 10 minutes")
            if sd.n >= 2:
                series.append(sd)
        except (KeyError, WindowError) as exc:
            errors.append(str(exc))
    res = rank_rates(series)
    res["window"] = window or "last 10 minutes"
    if res["ranking"]:
        ctx.focus["metric"] = res["ranking"][0]["metric"]
    return res


def t_anomalies(ctx: ToolContext, metric: str, window: str | None = None, source: str | None = None, threshold: float = 3.5) -> dict[str, Any]:
    sd, win, meta = _series(ctx, metric, window, source)
    res = {**meta, **detect_anomalies(sd, threshold=threshold, window=None if sd.n < 60 else 30)}
    if sd.n >= 3:
        res["peaks"] = detect_peaks(sd)["peaks"][-10:]
    if res.get("anomalies"):
        last = res["anomalies"][-1]
        ctx.focus["last_anomaly"] = {"metric": sd.metric, **last}
    ctx.service.db.save_analysis("anomalies", {"metric": sd.metric, "window": win.to_dict()}, res)
    return res


def t_compare_sources(ctx: ToolContext, metric: str, source_a: str | None = None, source_b: str | None = None, window: str | None = None) -> dict[str, Any]:
    resolved = ctx.service.resolve_metric(metric)
    if not resolved:
        raise ToolError(f"No data for metric '{metric}'.")
    cands = ctx.service.sources_for(resolved)
    ids = list(dict.fromkeys(c["source_id"] for c in cands))
    if len(ids) < 2 and not (source_a and source_b):
        return {"metric": resolved, "sources": ids, "note": "Only one source has this metric, so there is nothing to compare."}
    a_id = source_a or ids[0]
    b_id = source_b or next(i for i in ids if i != a_id)
    a, _, _ = ctx.service.get_series(resolved, window or "all", a_id)
    b, _, _ = ctx.service.get_series(resolved, window or "all", b_id)
    return compare_sources(a, b)


def t_correlate(ctx: ToolContext, metric_a: str, metric_b: str, window: str | None = None) -> dict[str, Any]:
    a, win, _ = ctx.service.get_series(metric_a, window or "all")
    b, _, _ = ctx.service.get_series(metric_b, win)
    res = correlation(a, b)
    res["lag"] = lagged_correlation(a, b)
    res["window"] = win.to_dict()
    return res


def t_explain_change(ctx: ToolContext, metric: str, window: str | None = None) -> dict[str, Any]:
    """Evidence for 'why did X change': where the change happened, what the
    other metrics did just before it, and how strongly they co-moved."""
    sd, win, meta = _series(ctx, metric, window or "last 30 minutes")
    if sd.n < 5:
        raise InsufficientData(f"Need at least 5 observations of '{sd.metric}' to examine a change; found {sd.n}.")
    s = sd.series
    diffs = s.diff().abs()
    t_change = diffs.idxmax()
    before = TimeWindow(max(s.index[0], t_change - (t_change - s.index[0]) / 2).to_pydatetime(), t_change.to_pydatetime(), "before the largest step")
    res: dict[str, Any] = {
        **meta,
        "largest_step": {"at": t_change.isoformat(), "step": float(s.diff().loc[t_change]), "value_after": float(s.loc[t_change])},
        "statistics": descriptive(sd),
        "related_metrics": [],
        "caution": "Associations below are not proof of cause. Physical causes need domain confirmation.",
    }
    for other in ctx.service.metric_names():
        if other == sd.metric:
            continue
        try:
            ob, _, _ = ctx.service.get_series(other, win)
        except (KeyError, WindowError):
            continue
        if ob.n < 3:
            continue
        corr = correlation(sd, ob)
        pre = ob.series.loc[before.start : before.end]
        moved = None
        if pre.size >= 2:
            moved = float(pre.iloc[-1] - pre.iloc[0])
        res["related_metrics"].append({"metric": other, "pearson_r": corr["pearson_r"], "paired_samples": corr["paired_samples"], "change_before_step": moved, "unit": ob.unit, "status": ob.status})
    res["related_metrics"].sort(key=lambda r: abs(r["pearson_r"] or 0), reverse=True)
    res["related_metrics"] = res["related_metrics"][:8]
    ctx.focus["last_event"] = {"metric": sd.metric, "timestamp": t_change.isoformat()}
    return res


def t_plot(ctx: ToolContext, metrics: list[str], window: str | None = None, source: str | None = None, show_anomalies: bool = True, title: str | None = None) -> dict[str, Any]:
    if not metrics:
        raise ToolError("Give at least one metric to plot.")
    series: list[Any] = []
    anomalies: dict[str, list[dict[str, Any]]] = {}
    thresholds: dict[str, list[tuple[str, float]]] = {}
    used: list[dict[str, Any]] = []
    skipped: list[str] = []
    win: TimeWindow | None = None
    for m in metrics[:6]:
        try:
            sd, w, meta = ctx.service.get_series(m, window or "all", source)
        except KeyError as exc:
            skipped.append(str(exc))
            continue
        win = win or w
        if sd.n == 0:
            skipped.append(f"No data for {sd.metric} in {w.describe()}.")
            continue
        series.append(sd)
        used.append({"metric": sd.metric, "source_id": meta["source_id"], "status": sd.status, "n": sd.n, "window": w.describe()})
        if show_anomalies and sd.n >= 5:
            anomalies[sd.metric] = detect_anomalies(sd)["anomalies"]
        for rule in ctx.alerts.rules_for(sd.metric):
            if rule.kind in ("above", "below") and rule.threshold is not None:
                thresholds.setdefault(sd.metric, []).append((rule.kind, rule.threshold))
            if rule.kind == "outside" and rule.threshold is not None and rule.lower is not None:
                thresholds.setdefault(sd.metric, []).extend([("upper", rule.threshold), ("lower", rule.lower)])
    if not series:
        raise ToolError("Nothing to plot. " + " ".join(skipped))
    mas = {sd.metric: moving_average(sd, max(3, sd.n // 20)) for sd in series if sd.n >= 20}
    fig = plots.trend_figure(series, title or ", ".join(s.metric for s in series), anomalies, thresholds, mas)
    desc = {"series": used, "skipped": skipped, "anomaly_points": {k: len(v) for k, v in anomalies.items()}}
    fid = ctx.service.add_figure(json.loads(fig.to_json()), desc)
    ctx.focus.update(figure_id=fid, metric=series[0].metric, plotted=[s.metric for s in series])
    return {"figure_id": fid, **desc, "note": "The figure is displayed to the user under your answer."}


def t_alerts(ctx: ToolContext) -> dict[str, Any]:
    active = ctx.service.db.alerts(active_only=True)
    return {"active_alerts": active[:20], "rules": [r.to_dict() | {"description": r.describe()} for r in ctx.alerts.rules.values()], "recent": ctx.service.db.alerts(limit=10)}


def t_record_recommendation(ctx: ToolContext, text: str, metric: str, direction: str, target: float | None = None, lower: float | None = None, window_minutes: float = 10) -> dict[str, Any]:
    resolved = ctx.service.resolve_metric(metric)
    if not resolved:
        raise ToolError(f"No data for metric '{metric}', so a measurable criterion cannot be set.")
    crit = recs.make_criteria(resolved, direction, target, lower, window_minutes * 60)
    sd, win, meta = ctx.service.get_series(resolved, f"last {int(max(window_minutes, 1))} minutes")
    evidence = {"statistics": descriptive(sd) if sd.n else None, "window": win.to_dict(), "source_id": meta["source_id"]}
    rid = ctx.service.db.add_recommendation(ctx.conversation_id, resolved, meta["source_id"], win.to_dict(), evidence, text, crit)
    ctx.focus["last_recommendation_id"] = rid
    return {"recommendation_id": rid, "criterion": recs.describe_criteria(crit)}


def t_evaluate_recommendation(ctx: ToolContext, recommendation_id: str | None = None) -> dict[str, Any]:
    items = ctx.service.db.recommendations(limit=50)
    if recommendation_id:
        items = [r for r in items if r["id"] == recommendation_id]
    elif ctx.conversation_id:
        conv = [r for r in items if r["conversation_id"] == ctx.conversation_id]
        items = conv or items
    if not items:
        return {"note": "No recommendation has been recorded yet, so there is nothing to evaluate."}
    rec = items[0]
    res = recs.evaluate(ctx.service.db, rec)
    ctx.service.db.set_recommendation_outcome(rec["id"], res["status"], res)
    res["recommendation_text"] = rec["text"][:500]
    res["recorded_at"] = rec["created_at"]
    return res


def t_read_chart_table(ctx: ToolContext, chart_id: str | None = None) -> dict[str, Any]:
    """Latest chart-model chart-to-table readings, with the cross-check against the pixel reader."""
    data = ctx.service.chart_model
    if not data:
        enabled = ctx.chart_model is not None
        return {"available": False, "note": "No chart-model reading yet." + ("" if enabled else " The chart model is not enabled in Settings.")}
    readings = data["readings"]
    if chart_id:
        readings = [r for r in readings if r["chart_id"] == chart_id] or readings
    age = (utcnow() - pd.Timestamp(data["captured_at"]).to_pydatetime()).total_seconds()
    if readings:
        ctx.focus["chart_id"] = readings[0]["chart_id"]
    return {
        "available": True,
        "frame_seq": data["frame_seq"],
        "captured_at": data["captured_at"],
        "age_seconds": round(age, 1),
        "readings": readings,
        "caution": "Chart-model values are a model's reading of the chart image (estimates). Agreement with the pixel reader raises confidence; disagreement means at least one reading is wrong.",
    }


def t_ask_chart_model(ctx: ToolContext, question: str, chart_id: str | None = None) -> dict[str, Any]:
    """Ask the chart model a question about a chart in the latest captured frame."""
    from vision.chart_model import ChartModelError, crop_for_chart

    if ctx.chart_model is None or ctx.latest_image is None:
        raise ToolError("The chart model is not enabled. Turn it on in Settings and start its server.")
    img = ctx.latest_image()
    if img is None:
        raise ToolError("No screen has been captured yet.")
    analysis = ctx.service.latest_screen
    chart = None
    if analysis is not None and analysis.charts:
        chart = next((c for c in analysis.charts if c.id == (chart_id or ctx.focus.get("chart_id"))), analysis.charts[0])
    try:
        res = ctx.chart_model.chart_qa(crop_for_chart(img, chart), question)
    except ChartModelError as exc:
        return {"error": str(exc), "kind": exc.kind}
    return {
        "question": question,
        "chart_id": chart.id if chart else "whole frame",
        "chart_title": chart.title if chart else "",
        "chart_model_answer": res.get("text", "")[:1500],
        "inference_ms": res.get("ms"),
        "status": "model answer about the chart image (estimate, not a measured value)",
    }


def t_report(ctx: ToolContext, window: str | None = None) -> dict[str, Any]:
    from core.reporting import build_report

    md = build_report(ctx.service, ctx.alerts, window or "all")
    rid = ctx.service.add_report(md)
    ctx.focus["report_id"] = rid
    return {"report_id": rid, "characters": len(md), "preview": md[:1500], "note": "The user can download the full report under your answer."}


def default_registry() -> ToolRegistry:
    W = Param("window", "string", "Time window, e.g. 'last 10 minutes', 'previous 10 minutes', '1h', 'today', 'all'.")
    M = Param("metric", "string", "Metric name as listed by list_metrics.", required=True)
    S = Param("source", "string", "Optional source id (from list_metrics) to force a specific source.")
    reg = ToolRegistry()
    reg.register(Tool("get_latest_screen_context", "Latest analyzed screen capture: charts, axes, series estimates, KPI tiles, capture time and age.", [], t_screen))
    reg.register(Tool("list_detected_charts", "Charts detected in the latest screen capture with series and confidence.", [], t_list_charts))
    reg.register(Tool("list_metrics", "All metrics with sources, units, counts, time ranges and whether values are verified or estimated.", [], t_list_metrics))
    reg.register(Tool("get_metric_history", "Timestamped values of one metric in a window (downsampled).", [M, W, S, Param("max_points", "integer", "Maximum points to return.", minimum=2, maximum=200)], t_history))
    reg.register(Tool("calculate_statistics", "Descriptive statistics, rate of change, trend test and data quality for a metric in a window.", [M, W, S], t_stats))
    reg.register(
        Tool(
            "compare_time_windows",
            "Compare a metric between two windows (means, extremes, % change, Mann-Whitney test).",
            [M, Param("window_a", "string", "Earlier window, e.g. 'previous 10 minutes'."), Param("window_b", "string", "Later window, e.g. 'last 10 minutes'."), S],
            t_compare_windows,
        )
    )
    reg.register(Tool("rank_rates_of_change", "Rank all metrics by how fast they change in a window (relative to their level).", [W], t_rank_rates))
    reg.register(Tool("detect_anomalies", "Robust z-score anomalies and peaks for a metric.", [M, W, S, Param("threshold", "number", "Robust z threshold (default 3.5).", minimum=1, maximum=10)], t_anomalies))
    reg.register(Tool("compare_data_sources", "Compare the same metric across two sources (e.g. screen estimate vs file) and list discrepancies.", [M, Param("source_a", "string", "First source id."), Param("source_b", "string", "Second source id."), W], t_compare_sources))
    reg.register(Tool("correlate_metrics", "Correlation (Pearson/Spearman) and lead/lag between two metrics. Association only.", [Param("metric_a", "string", "First metric.", required=True), Param("metric_b", "string", "Second metric.", required=True), W], t_correlate))
    reg.register(Tool("explain_change", "Evidence for why a metric changed: largest step, what other metrics did before it, correlations.", [M, W], t_explain_change))
    reg.register(
        Tool(
            "generate_trend_plot",
            "Create an interactive graph of one or more metrics; it is shown to the user.",
            [Param("metrics", "array", "Metric names to plot.", required=True, max_items=6), W, S, Param("show_anomalies", "boolean", "Mark anomalies."), Param("title", "string", "Optional title.", max_length=80)],
            t_plot,
        )
    )
    reg.register(Tool("get_active_alerts", "Active alerts, alert rules and recent alert history.", [], t_alerts))
    reg.register(
        Tool(
            "record_recommendation",
            "Record a recommendation with a measurable success criterion so it can be evaluated later.",
            [
                Param("text", "string", "The recommendation.", required=True, max_length=600),
                M,
                Param("direction", "string", "Success criterion.", required=True, enum=list(recs.VALID_DIRECTIONS)),
                Param("target", "number", "Target / upper bound for below/above/within."),
                Param("lower", "number", "Lower bound for within."),
                Param("window_minutes", "number", "Evaluation window in minutes.", minimum=0.5, maximum=1440),
            ],
            t_record_recommendation,
        )
    )
    reg.register(Tool("evaluate_recommendation", "Check the latest (or given) recommendation's success criterion using before/after measurements.", [Param("recommendation_id", "string", "Optional id.")], t_evaluate_recommendation))
    reg.register(Tool("read_chart_table", "Latest chart-model (Granite Vision / ChartAst) chart-to-table readings of the screen charts, cross-checked against the pixel reader.", [Param("chart_id", "string", "Optional chart id, e.g. chart1.")], t_read_chart_table))
    reg.register(
        Tool(
            "ask_chart_model",
            "Ask the chart model (Granite Vision / ChartAst) a question about a chart in the latest screen capture (e.g. a value at a label, which series is highest). Answers are estimates.",
            [Param("question", "string", "Question about the chart.", required=True, max_length=400), Param("chart_id", "string", "Optional chart id.")],
            t_ask_chart_model,
        )
    )
    reg.register(Tool("generate_analysis_report", "Build a downloadable Markdown report of metrics, statistics, anomalies, alerts and recommendations.", [W], t_report))
    return reg


def window_minutes(td: timedelta) -> float:
    return td.total_seconds() / 60


def history_to_frame(points: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(points)
