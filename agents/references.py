"""Conversation state and reference resolution.

Keeps a small structured "focus" (current metric, chart, window, last anomaly,
last figure, last recommendation) that tools update, so follow-ups such as
"this graph", "that spike", "the previous value", "what happened afterward"
and "did your earlier recommendation work?" resolve to concrete evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from analysis.windows import parse_duration
from core.services import DataService

REF_PATTERNS = {
    "this_chart": re.compile(r"\b(this|that|the|current)\s+(graph|chart|plot|figure|panel)\b", re.I),
    "both_charts": re.compile(r"\b(both|two|all)\s+(graphs|charts|plots|panels)\b|\bcompare (the|both) charts\b", re.I),
    "that_spike": re.compile(r"\b(that|the|this)\s+(spike|peak|jump|drop|dip|anomal\w*|outlier|surge)\b", re.I),
    "previous_value": re.compile(r"\b(previous|last|prior|earlier)\s+(value|reading|measurement)\b", re.I),
    "afterward": re.compile(r"\b(afterwards?|after that|after it|since then|what happened (next|after))\b", re.I),
    "earlier_recommendation": re.compile(r"\b(your|the|that|last|earlier|previous)\s+(suggestion|recommendation|advice)\b|\bdid (it|that) (work|help)\b", re.I),
    "it_metric": re.compile(r"\b(it|this metric|that metric|this value|that value)\b", re.I),
}


@dataclass
class ConversationState:
    conversation_id: str
    focus: dict[str, Any] = field(default_factory=dict)
    turns: int = 0

    def update_focus(self, changes: dict[str, Any]) -> None:
        self.focus.update({k: v for k, v in changes.items() if v is not None})


@dataclass
class Resolution:
    notes: list[str]
    metric: str | None
    window: str | None
    refs: set[str]
    event_time: datetime | None = None


def mentioned_metrics(question: str, service: DataService) -> list[str]:
    """Metrics named in the question (exact substring first, then fuzzy on words)."""
    q = question.lower()
    names = service.metric_names()
    hits = [n for n in sorted(names, key=len, reverse=True) if re.search(rf"\b{re.escape(n.lower())}\b", q)]
    if hits:
        # drop names contained in longer hits ("current" inside "current a")
        return [h for h in hits if not any(h != o and h.lower() in o.lower() for o in hits)]
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", q)
    out = []
    for w in words:
        r = service.resolve_metric(w)
        if r and r not in out and len(w) >= 4:
            out.append(r)
    return out[:4]


def window_phrase(question: str) -> str | None:
    q = question.lower()
    m = re.search(r"\b(last|past|previous|prior|preceding|over the|in the|during the)\s+[\w\s.-]{0,20}?(seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b", q)
    if m:
        return m.group(0)
    m2 = re.search(r"\b\d+(?:\.\d+)?\s*(s|m|min|h|hr|d)\b", q)
    if m2:
        return m2.group(0)
    if "today" in q:
        return "today"
    if re.search(r"\b(all (data|time)|entire|whole (file|dataset))\b", q):
        return "all"
    return None


def resolve_references(question: str, state: ConversationState, service: DataService) -> Resolution:
    refs = {name for name, pat in REF_PATTERNS.items() if pat.search(question)}
    notes: list[str] = []
    f = state.focus
    metrics = mentioned_metrics(question, service)
    metric = metrics[0] if metrics else None
    window = window_phrase(question)
    event_time = None
    if "this_chart" in refs:
        if f.get("figure_id") and f.get("plotted"):
            notes.append(f"'this graph' → the figure shown last ({f['figure_id']}) of {', '.join(f['plotted'])}.")
            metric = metric or f["plotted"][0]
        else:
            sc = service.screen_context()
            if sc.get("available") and sc.get("charts"):
                c = next((c for c in sc["charts"] if c["id"] == f.get("chart_id")), sc["charts"][0])
                notes.append(f"'this graph' → screen chart {c['id']} '{c['title']}' captured {sc['captured_at']} ({sc['age_seconds']} s ago).")
                if c["series"]:
                    metric = metric or service.resolve_metric(c["series"][0]["name"])
            elif not metrics:
                notes.append("'this graph' was mentioned but no chart has been shown or detected yet.")
    if "both_charts" in refs:
        sc = service.screen_context()
        charts = sc.get("charts", []) if sc.get("available") else []
        if len(charts) >= 2:
            notes.append(f"'both charts' → {charts[0]['id']} '{charts[0]['title']}' and {charts[1]['id']} '{charts[1]['title']}'.")
        elif f.get("plotted") and len(f["plotted"]) >= 2:
            notes.append(f"'both charts' → plotted metrics {f['plotted'][0]} and {f['plotted'][1]}.")
        else:
            notes.append("'both charts' was mentioned but fewer than two charts are available.")
    if "that_spike" in refs:
        an = f.get("last_anomaly") or f.get("last_event")
        if an:
            event_time = pd.Timestamp(an["timestamp"]).to_pydatetime()
            notes.append(f"'that spike' → {an['metric']} at {an['timestamp']}" + (f" (value {an['value']:.4g})" if "value" in an else "") + ".")
            metric = metric or an["metric"]
        else:
            notes.append("'that spike' was mentioned but no anomaly has been identified yet in this conversation; run anomaly detection first.")
    if "afterward" in refs:
        an = f.get("last_anomaly") or f.get("last_event")
        if an:
            event_time = pd.Timestamp(an["timestamp"]).to_pydatetime()
            dur = parse_duration(question) or None
            notes.append(f"'afterward' → from {an['timestamp']} onward" + (f" for {dur}" if dur else " until the latest data") + ".")
            metric = metric or an["metric"]
            window = window or f"{event_time.isoformat()}/{(event_time + (dur or timedelta(days=3650))).isoformat()}"
    if "earlier_recommendation" in refs:
        if f.get("last_recommendation_id"):
            notes.append(f"'your suggestion' → recommendation {f['last_recommendation_id']}.")
        else:
            recs = service.db.recommendations(limit=1, conversation_id=state.conversation_id) or service.db.recommendations(limit=1)
            notes.append(f"'your suggestion' → recommendation {recs[0]['id']} from {recs[0]['created_at'][:19]}." if recs else "'your suggestion' was mentioned but no recommendation has been recorded.")
    if "previous_value" in refs and (metric or f.get("metric")):
        m = metric or f["metric"]
        try:
            sd, _, meta = service.get_series(m, "all")
            if sd.n >= 2:
                notes.append(f"'previous value' of {sd.metric} → {sd.series.iloc[-2]:.6g} at {sd.series.index[-2].isoformat()} (latest {sd.series.iloc[-1]:.6g} at {sd.series.index[-1].isoformat()}).")
        except KeyError:
            pass
    if metric is None and f.get("metric") and ("it_metric" in refs or refs or not metrics):
        metric = f["metric"]
        if "it_metric" in refs or refs:
            notes.append(f"'it' / the current topic → metric {metric}.")
    if window is None and f.get("window") and refs & {"this_chart", "that_spike", "previous_value"}:
        window = None  # keep tools' defaults; window carried via focus only when explicit
    return Resolution(notes, metric, window, refs, event_time)
