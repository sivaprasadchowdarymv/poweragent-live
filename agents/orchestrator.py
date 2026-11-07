"""
Query orchestrator for live data (RAG∞ Ultimate design: deterministic, ~1 ms, no LLM call).

1. Resolve references against the conversation focus ("this graph", "that spike",
   "the previous value", "did your suggestion work?").
2. Classify intents with regex rules and pick ONLY the analysis tools needed
   (the "specialists"), so every number in the answer comes from Python.
3. Decide the answer route:
     use_tools  ReAct loop (the master may call more analysis tools) only for
                why / compare / correlate / advice questions — otherwise one call
     visual     the question is about what the screen or a chart LOOKS like, so
                chart crops are shown to the vision model (Gemma / Qwen)
4. Mark follow-ups (never served from the answer cache).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from agents.references import ConversationState, Resolution, mentioned_metrics, resolve_references
from core.services import DataService

INTENTS: list[tuple[str, re.Pattern[str]]] = [
    ("evaluate_recommendation", re.compile(r"\b(did|has|have) (it|that|your|the|my)?\s*(last |earlier |previous )?(suggestion|recommendation|advice|change|fix)?\s*(work|help|improve|fix)|\bdid it work\b|\bwas (it|that) effective\b", re.I)),
    ("screen", re.compile(r"\b(on (my|the) screen|what('?s| is) (happening|shown|showing|visible|going on)|what do you see|describe (the )?(screen|dashboard)|read (the )?(screen|dashboard))\b", re.I)),
    ("fastest", re.compile(r"\b(fastest|quickest|most rapidly|changing (the )?most|biggest change|which metric.*chang)\b", re.I)),
    ("why", re.compile(r"\b(why|what caused|cause of|reason for|explain (the|this|that)? ?(increase|decrease|rise|drop|change|spike))\b", re.I)),
    ("compare_windows", re.compile(r"\b(compare|versus|vs\.?|compared (to|with)|than (the )?(previous|last|earlier|before))\b.*\b(minutes?|hours?|seconds?|days?|window|period|before|earlier|previous)\b", re.I)),
    ("compare_sources", re.compile(r"\b(compare|discrepanc\w*|differ\w*|mismatch|agree)\b.*\b(sources?|file|screen|csv|excel|spreadsheet)\b|\bscreen (vs|versus) (file|csv)\b", re.I)),
    ("plot", re.compile(r"\b(show|plot|graph|draw|chart|visuali[sz]e)\b.*\b(graph|chart|plot|trend|me|it|history|over time|metric|data)\b|\bshow me\b", re.I)),
    ("suggest", re.compile(r"\b(what should (i|we)|recommend\w*|suggest\w*|advice|next steps?|what (can|do) (i|we) do|how (can|do) (i|we) (fix|improve|reduce))\b", re.I)),
    ("anomalies", re.compile(r"\b(anomal\w*|outliers?|spikes?|unusual|abnormal|glitch|surges?|peaks?)\b", re.I)),
    ("alerts", re.compile(r"\b(alerts?|alarms?|thresholds?|violations?|warnings?)\b", re.I)),
    ("report", re.compile(r"\b(report|export|summary document|write[- ]?up)\b", re.I)),
    ("correlate", re.compile(r"\b(correlat\w*|relationship between|related to|associated with)\b", re.I)),
    ("chart_question", re.compile(r"\b(according to|on|in|from) (the|this|that) (chart|graph|plot)\b|\b(read|ask) (the )?chart\b", re.I)),
    ("stats", re.compile(r"\b(average|mean|median|max(imum)?|min(imum)?|std|deviation|statistics|stats|how (high|low|much)|current value|latest value|value of|range|trend|increas\w*|decreas\w*|rate)\b", re.I)),
    ("metrics", re.compile(r"\b(what (data|metrics)|which metrics|list (the )?metrics|what can you see in the (file|data)|available data)\b", re.I)),
]


@dataclass
class Plan:
    intents: list[str]
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    clarification: str | None = None


def detect_intents(question: str) -> list[str]:
    return [name for name, pat in INTENTS if pat.search(question)]


def make_plan(question: str, res: Resolution, service: DataService, metrics_in_question: list[str], chart_model_enabled: bool = False) -> Plan:
    intents = detect_intents(question)
    if "earlier_recommendation" in res.refs and "evaluate_recommendation" not in intents:
        intents.insert(0, "evaluate_recommendation")
    if "evaluate_recommendation" in intents:
        # "did your suggestion work?" is an evaluation, not a request for new advice
        intents = [i for i in intents if i != "suggest"]
    plan = Plan(intents)
    metric = res.metric
    window = res.window
    has_data = bool(service.metric_names())
    has_screen = bool(service.screen_context().get("available"))
    add = plan.calls.append
    w = {"window": window} if window else {}

    if not has_data and not has_screen:
        plan.clarification = "No screen capture or data file has been analyzed yet."
        return plan

    for intent in intents:
        if intent == "screen":
            add(("get_latest_screen_context", {}))
            if service.chart_model:
                add(("read_chart_table", {}))
        elif intent == "fastest":
            add(("rank_rates_of_change", w))
        elif intent == "why" and metric:
            add(("explain_change", {"metric": metric, **w}))
        elif intent == "compare_windows" and metric:
            dur = _duration_phrase(question)
            a = f"previous {dur}" if dur else "previous 10 minutes"
            b = f"last {dur}" if dur else "last 10 minutes"
            add(("compare_time_windows", {"metric": metric, "window_a": a, "window_b": b}))
        elif intent == "compare_sources" and metric:
            add(("compare_data_sources", {"metric": metric, **w}))
        elif intent == "plot":
            targets = metrics_in_question or ([metric] if metric else [])
            if not targets and "both_charts" in res.refs:
                targets = service.metric_names()[:2]
            if not targets:
                targets = service.metric_names()[:3]
            if targets:
                add(("generate_trend_plot", {"metrics": targets, **w}))
        elif intent == "suggest":
            add(("get_active_alerts", {}))
            if metric:
                add(("calculate_statistics", {"metric": metric, **w}))
                add(("detect_anomalies", {"metric": metric, **w}))
            else:
                add(("rank_rates_of_change", w))
        elif intent == "anomalies" and metric:
            add(("detect_anomalies", {"metric": metric, **w}))
        elif intent == "alerts":
            add(("get_active_alerts", {}))
        elif intent == "report":
            add(("generate_analysis_report", w))
        elif intent == "correlate" and len(metrics_in_question) >= 2:
            add(("correlate_metrics", {"metric_a": metrics_in_question[0], "metric_b": metrics_in_question[1], **w}))
        elif intent == "stats" and metric:
            for m in metrics_in_question or [metric]:
                add(("calculate_statistics", {"metric": m, **w}))
        elif intent == "chart_question":
            if service.chart_model is not None or chart_model_enabled:
                add(("read_chart_table", {}))
            if chart_model_enabled:
                add(("ask_chart_model", {"question": question[:400]}))
        elif intent == "evaluate_recommendation":
            add(("evaluate_recommendation", {}))
        elif intent == "metrics":
            add(("list_metrics", {}))

    needs_metric = {"why", "compare_windows", "compare_sources", "anomalies", "stats"} & set(intents)
    if needs_metric and not metric and has_data:
        names = service.metric_names()
        if len(names) == 1:
            metric = names[0]
            return make_plan(question + f" ({metric})", Resolution(res.notes, metric, res.window, res.refs, res.event_time), service, [metric], chart_model_enabled)
        plan.clarification = f"Which metric do you mean? Available: {', '.join(names[:12])}."
    if not plan.calls:
        # General question: give the model an overview to ground the answer.
        if has_screen:
            add(("get_latest_screen_context", {}))
        if has_data:
            add(("list_metrics", {}))
            if metric:
                add(("calculate_statistics", {"metric": metric, **w}))
    # de-duplicate while keeping order
    seen, uniq = set(), []
    for name, args in plan.calls:
        key = (name, tuple(sorted((k, str(v)) for k, v in args.items())))
        if key not in seen:
            seen.add(key)
            uniq.append((name, args))
    plan.calls = uniq[:6]
    return plan


def _duration_phrase(q: str) -> str | None:
    m = re.search(r"\b(\d+(?:\.\d+)?|one|two|three|five|ten|fifteen|twenty|thirty|sixty)\s*(seconds?|minutes?|mins?|hours?|hrs?|days?)\b", q, re.I)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m2 = re.search(r"\b(previous|last|past)\s+(minute|hour|day)\b", q, re.I)
    if m2:
        return f"1 {m2.group(2)}"
    return None


# ---------------------------------------------------------------------------- routing
_TOOL_INTENTS = {"why", "compare_windows", "compare_sources", "correlate", "suggest", "fastest"}
_VISUAL = re.compile(
    r"\b(what do you see|look(s)? like|describe (the )?(screen|dashboard|chart|graph)|on (my|the) screen|"
    r"(colou?r|legend|label|annotation|marker|shape|pattern|curve|line|bar)s?\b.*\b(chart|graph|plot|screen)|"
    r"(chart|graph|plot)\b.*\b(show|shows|showing|look|read|say|says)|read (the |this )?(chart|graph|screen|dashboard)|"
    r"according to (the|this|that) (chart|graph|plot|screen)|which (series|line|bar) is)\b", re.I)
# Any mention of the screen or a picture-like thing sends the question to the vision model.
_SCREENISH = re.compile(r"\b(screen|dashboard|chart|charts|graph|graphs|plot|plots|image|picture|figure|curve|trend|"
                        r"waveform|display|panel|gauge|this|that|see|visible|shown|showing|look)\b", re.I)
_FOLLOW_UP = re.compile(r"^(and|what about|how about|also|then|same|but)\b", re.I)
_PRONOUN = re.compile(r"\b(it|its|that|this|those|these|they|them|the same)\b", re.I)


@dataclass
class RoutePlan:
    intents: list[str]
    calls: list[tuple[str, dict[str, Any]]]
    clarification: str | None
    resolution: Resolution
    metrics_in_question: list[str]
    use_tools: bool = False
    visual: bool = False
    is_follow_up: bool = False
    latency_ms: float = 0.0
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"intents": self.intents, "calls": [n for n, _ in self.calls], "use_tools": self.use_tools,
                "visual": self.visual, "follow_up": self.is_follow_up, "metric": self.resolution.metric,
                "window": self.resolution.window, "notes": self.resolution.notes}


def route(question: str, state: ConversationState, service: DataService, previous_question: str | None = None,
          agent_mode: str = "agent", chart_model_enabled: bool = False, screen_available: bool = False) -> RoutePlan:
    t0 = time.perf_counter()
    q = question.strip()
    res = resolve_references(q, state, service)
    metrics_q = mentioned_metrics(q, service)
    plan = make_plan(q, res, service, metrics_q, chart_model_enabled=chart_model_enabled)
    follow_up = bool(previous_question) and (bool(_FOLLOW_UP.match(q)) or bool(res.refs)
                                             or (len(q.split()) <= 10 and bool(_PRONOUN.search(q))))
    # Vision whenever a capture exists and the question is about the screen/charts, or when no file
    # data exists at all (then the screen IS the data and the pixel reader alone is often too thin).
    # Vision-first: whenever an image/capture exists the vision model looks at it, except for questions
    # that only name file metrics and say nothing about the screen.
    file_only = bool(metrics_q) and not (_VISUAL.search(q) or _SCREENISH.search(q))
    visual = screen_available and not file_only
    use_tools = agent_mode == "agent" and bool(_TOOL_INTENTS & set(plan.intents))
    return RoutePlan(plan.intents, plan.calls, plan.clarification, res, metrics_q, use_tools, visual, follow_up,
                     (time.perf_counter() - t0) * 1000)
