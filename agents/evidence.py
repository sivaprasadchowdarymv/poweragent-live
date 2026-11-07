"""
Evidence registry: every computed result gets a stable, citable label (RAG∞ Ultimate design).

    [REF-1: SCREEN frame 42]          latest analyzed screen capture (OCR + pixel chart reader)
    [REF-2: STATS current_a, last 10 minutes]
    [REF-3: ANOMALY current_a]
    [REF-4: SCREEN-IMAGE chart1]      a chart crop that was SHOWN to the vision model

The model sees a compact text rendering of each result (bounded by MAX_CONTEXT_CHARS);
the verifier checks the answer's numbers against the FULL result data, so nothing is
lost to truncation on the checking side.

Screen text, chart titles and file contents are untrusted: every rendering passes the
prompt-injection guard (ops.guard.sanitize_evidence) before it reaches a model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agents.tools import ToolCallRecord
from ops.guard import sanitize_evidence

KIND = {
    "get_latest_screen_context": "SCREEN",
    "list_detected_charts": "SCREEN",
    "list_metrics": "METRICS",
    "get_metric_history": "HISTORY",
    "calculate_statistics": "STATS",
    "compare_time_windows": "WINDOWS",
    "rank_rates_of_change": "RATES",
    "detect_anomalies": "ANOMALY",
    "compare_data_sources": "SOURCES",
    "correlate_metrics": "CORR",
    "explain_change": "CHANGE",
    "generate_trend_plot": "PLOT",
    "get_active_alerts": "ALERTS",
    "record_recommendation": "REC",
    "evaluate_recommendation": "REC-CHECK",
    "read_chart_table": "CHART-MODEL",
    "ask_chart_model": "CHART-MODEL",
    "generate_analysis_report": "REPORT",
}
# Keys that are noise for the model (ids, colours, internal bookkeeping).
_SKIP_KEYS = {"color", "frame_seq", "image_size", "figure", "vision_model_notes_note", "chart_model_note"}
_MAX_LIST = 10
_MAX_STR = 240


@dataclass(eq=False)
class SourceRef:
    """One citable evidence item. `data` is the full result (used by the verifier)."""

    ref_num: int
    label: str
    kind: str
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    ok: bool = True
    visual: bool = False  # an image shown to the vision model
    data: Any = None

    def as_dict(self) -> Dict[str, Any]:
        """Serializable form kept with the chat message (full data is not stored)."""
        return {"ref_num": self.ref_num, "label": self.label, "kind": self.kind, "tool": self.tool,
                "args": self.args, "text": self.text[:1500], "ok": self.ok, "visual": self.visual}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceRef":
        return cls(int(d.get("ref_num", 0)), str(d.get("label", "")), str(d.get("kind", "")), str(d.get("tool", "")),
                   dict(d.get("args") or {}), str(d.get("text", "")), bool(d.get("ok", True)), bool(d.get("visual", False)))


def _fmt_num(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}"


def compact(obj: Any, prefix: str = "", out: Optional[List[str]] = None, depth: int = 0) -> List[str]:
    """Flatten a result into short 'path: value' lines (lists truncated, strings bounded)."""
    out = out if out is not None else []
    if depth > 6:
        return out
    if isinstance(obj, dict):
        scalars = []
        for k, v in obj.items():
            if k in _SKIP_KEYS or v is None or v == [] or v == {}:
                continue
            if isinstance(v, (dict, list)):
                continue
            scalars.append(f"{k}={_scalar(v)}")
        if scalars:
            out.append((f"{prefix}: " if prefix else "") + ", ".join(scalars))
        for k, v in obj.items():
            if k in _SKIP_KEYS or not isinstance(v, (dict, list)) or not v:
                continue
            compact(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        items = obj[:_MAX_LIST]
        if all(not isinstance(x, (dict, list)) for x in items):
            more = f" (+{len(obj) - _MAX_LIST} more)" if len(obj) > _MAX_LIST else ""
            out.append(f"{prefix}: [{', '.join(_scalar(x) for x in items)}]{more}")
        else:
            for i, x in enumerate(items):
                compact(x, f"{prefix}[{i}]", out, depth + 1)
            if len(obj) > _MAX_LIST:
                out.append(f"{prefix}: (+{len(obj) - _MAX_LIST} more items not shown)")
    else:
        out.append(f"{prefix}: {_scalar(obj)}")
    return out


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _fmt_num(float(v))
    s = str(v).replace("\n", " ")
    return s if len(s) <= _MAX_STR else s[:_MAX_STR] + "…"


def _detail(rec: ToolCallRecord) -> str:
    a = rec.arguments or {}
    if rec.name in ("get_latest_screen_context", "list_detected_charts") and isinstance(rec.result, dict):
        seq = rec.result.get("frame_seq")
        return f"frame {seq}" if seq is not None else "latest capture"
    parts = []
    for key in ("metric", "metric_a", "metric_b", "metrics", "chart_id", "window", "window_b", "source"):
        if a.get(key):
            v = a[key]
            parts.append(", ".join(map(str, v)) if isinstance(v, list) else str(v))
    return ", ".join(parts[:3])


class EvidenceRegistry:
    """Gives every evidence item a stable [REF-n: KIND detail] label."""

    def __init__(self) -> None:
        self.refs: List[SourceRef] = []
        self.injection_lines = 0

    def _next(self) -> int:
        return len(self.refs) + 1

    def add_record(self, rec: ToolCallRecord) -> SourceRef:
        n = self._next()
        kind = KIND.get(rec.name, rec.name.upper())
        detail = _detail(rec)
        label = f"[REF-{n}: {kind}{' ' + detail if detail else ''}]"
        body = rec.result if rec.ok else {"error": (rec.result or {}).get("error", "failed")}
        text = "\n".join(compact(body))
        text, hits = sanitize_evidence(text)
        self.injection_lines += hits
        ref = SourceRef(n, label, kind, rec.name, dict(rec.arguments or {}), text, rec.ok, False, rec.result)
        self.refs.append(ref)
        return ref

    def add_note(self, kind: str, detail: str, text: str, data: Any = None, visual: bool = False) -> SourceRef:
        n = self._next()
        text, hits = sanitize_evidence(text)
        self.injection_lines += hits
        ref = SourceRef(n, f"[REF-{n}: {kind}{' ' + detail if detail else ''}]", kind, "", {}, text, True, visual,
                        data if data is not None else text)
        self.refs.append(ref)
        return ref

    def by_number(self, n: int) -> Optional[SourceRef]:
        return self.refs[n - 1] if 0 < n <= len(self.refs) else None

    def render(self, refs: Optional[Sequence[SourceRef]] = None, budget: int = 5000) -> str:
        """Evidence block for the prompt, fair-shared across items within `budget` characters."""
        items = list(refs if refs is not None else self.refs)
        if not items:
            return ""
        share = max(400, budget // len(items))
        blocks, used = [], 0
        for r in items:
            body = r.text if len(r.text) <= share else r.text[:share].rsplit("\n", 1)[0] + "\n… (truncated)"
            block = f"{r.label}\n{body}"
            if used + len(block) > budget and blocks:
                blocks.append(f"(… {len(items) - len(blocks)} more evidence item(s) omitted to stay within the budget)")
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    def to_json(self) -> str:
        return json.dumps([r.as_dict() for r in self.refs], default=str)
