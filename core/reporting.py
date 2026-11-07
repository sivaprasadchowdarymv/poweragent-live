"""Markdown analysis report built purely from stored, computed results."""

from __future__ import annotations

from typing import Any

from analysis.alerts import AlertEngine
from analysis.stats import InsufficientData, descriptive, detect_anomalies, trend
from core.models import utcnow
from core.services import DataService


def _f(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)


def build_report(service: DataService, alerts: AlertEngine, window: str = "all") -> str:
    now = utcnow()
    out = [
        "# PowerAgent Live Dashboard — Analysis Report",
        "",
        f"Generated {now.strftime('%Y-%m-%d %H:%M:%S')} UTC · window: **{window}**",
        "",
        "All statistics below were computed in Python from the stored observations. "
        "Values marked *estimated* were read from screenshots and are approximate.",
        "",
        "## Sources",
        "",
        "| Source | Type | Observations | First | Last |",
        "|---|---|---|---|---|",
    ]
    for s in service.db.sources():
        out.append(f"| {s['name']} (`{s['id']}`) | {s['type']} | {s['observations']:,} | {s['first'] or '—'} | {s['last'] or '—'} |")
    out += ["", "## Metrics", "", "| Metric | Source | Status | n | Mean | Min | Max | Last | Trend |", "|---|---|---|---|---|---|---|---|---|"]
    anomaly_lines: list[str] = []
    notes: list[str] = []
    for name in service.metric_names():
        try:
            sd, win, meta = service.get_series(name, window)
        except Exception as exc:  # keep the report going
            notes.append(f"- {name}: {exc}")
            continue
        if sd.n == 0:
            continue
        d = descriptive(sd)
        tr = "—"
        try:
            tr = trend(sd)["direction"] if sd.n >= 3 else "—"
        except InsufficientData:
            pass
        out.append(f"| {sd.metric} {('(' + sd.unit + ')') if sd.unit else ''} | `{meta['source_id']}` | {sd.status} | {sd.n:,} | {_f(d['mean'])} | {_f(d['min'])} | {_f(d['max'])} | {_f(d['last'])} | {tr} |")
        notes += [f"- {sd.metric}: {n}" for n in sd.notes]
        if sd.n >= 5:
            an = detect_anomalies(sd)
            for a in an["anomalies"][:10]:
                anomaly_lines.append(f"| {sd.metric} | {a['timestamp']} | {_f(a['value'])} | {a['robust_z']:.2f} | {a['direction']} |")
    out += ["", "## Anomalies (robust z ≥ 3.5)", ""]
    if anomaly_lines:
        out += ["| Metric | Time | Value | Robust z | Direction |", "|---|---|---|---|---|", *anomaly_lines]
    else:
        out.append("No anomalies detected.")
    out += ["", "## Alerts", ""]
    al = service.db.alerts(limit=50)
    if al:
        out += ["| Severity | Rule | Opened | Resolved | Occurrences |", "|---|---|---|---|---|"]
        out += [f"| {a['severity']} | {a['message']} | {a['opened_at']} | {a['resolved_at'] or 'active'} | {a['occurrences']} |" for a in al]
    else:
        out.append("No alerts recorded.")
    out += ["", "## Recommendations", ""]
    rs = service.db.recommendations(limit=20)
    if rs:
        for r in rs:
            out.append(f"- **{r['created_at'][:19]}** — {r['text'][:300]}  \n  Criterion: {r['criteria']['direction']} of {r['metric']} · status: {r['status']}")
    else:
        out.append("No recommendations recorded.")
    sc = service.screen_context()
    if sc.get("available"):
        out += ["", "## Latest screen analysis", "", f"Captured {sc['captured_at']} ({sc['age_seconds']} s before this report)."]
        for c in sc["charts"]:
            out.append(f"- {c['title']} ({c['type']}, confidence {c['confidence']}): " + ", ".join(f"{s['name']} ≈ {_f(s['latest_estimate'])}" for s in c["series"]))
        for k in sc["kpis"]:
            out.append(f"- KPI {k['label']} ≈ {_f(k['value_estimate'])} {k['unit']} (confidence {k['confidence']})")
    if notes:
        out += ["", "## Data-quality notes", "", *notes]
    return "\n".join(out) + "\n"
