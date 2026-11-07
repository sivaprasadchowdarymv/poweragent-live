"""Deterministic rendering of tool results.

Used when Ollama is offline or the model is missing, so the user still gets
the computed facts (clearly labelled as an automatic summary, not an AI answer).
"""

from __future__ import annotations

from typing import Any

from agents.tools import ToolCallRecord


def f(v: Any, unit: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        s = f"{v:,.4g}"
    else:
        s = str(v)
    return f"{s} {unit}".strip()


def render(records: list[ToolCallRecord]) -> str:
    lines: list[str] = []
    for r in records:
        res = r.result
        if "error" in res:
            lines.append(f"- **{r.name}**: {res['error']}")
            continue
        if r.name == "get_latest_screen_context":
            if not res.get("available"):
                lines.append(f"- Screen: {res.get('reason')}")
                continue
            lines.append(f"**Screen** captured {res['captured_at'][:19]} UTC ({res['age_seconds']} s ago) — values are estimates from the screenshot:")
            for c in res.get("charts", []):
                ser = "; ".join(f"{s['name']} ≈ {f(s['latest_estimate'], c['unit'])} (conf {s['confidence']})" for s in c["series"]) or "no series read"
                lines.append(f"- {c['title']} [{c['type']}]: {ser}")
            for k in res.get("kpis", []):
                lines.append(f"- KPI {k['label']} ≈ {f(k['value_estimate'], k['unit'])}")
        elif r.name == "calculate_statistics":
            s = res["statistics"]
            u = s["provenance"]["unit"]
            lines.append(
                f"**{s['provenance']['metric']}** ({s['provenance']['status']}, source `{res['source_id']}`, {res['window_description']}): "
                f"n={s['count']}, mean {f(s['mean'], u)}, min {f(s['min'], u)}, max {f(s['max'], u)}, last {f(s['last'], u)} at {s['last_at'][:19]}, change {f(s['change_pct'])} %."
            )
            if "trend" in res:
                lines.append(f"- Trend: {res['trend']['direction']} (Kendall p={f(res['trend']['p_value'])})")
        elif r.name == "rank_rates_of_change":
            lines.append("**Fastest-changing metrics** (relative rate, % of level per minute):")
            for row in res["ranking"][:5]:
                lines.append(f"- {row['metric']}: {f(row['relative_rate_pct'])} %/min ({f(row['slope'])} {row['rate_unit']}, {row['status']})")
        elif r.name == "compare_time_windows":
            keys = [k for k in res if isinstance(res[k], dict) and "mean" in res[k]]
            if len(keys) == 2:
                a, b = keys
                lines.append(f"**{a}** mean {f(res[a]['mean'])} vs **{b}** mean {f(res[b]['mean'])} → difference {f(res['mean_difference'])} ({f(res['mean_change_pct'])} %), Mann-Whitney p={f(res['significance']['p_value'])}.")
        elif r.name == "detect_anomalies":
            lines.append(f"**Anomalies** in {res.get('metric')}: {res['count']} found ({res['method']}).")
            for a in res["anomalies"][-5:]:
                lines.append(f"- {a['timestamp'][:19]}: {f(a['value'])} (z={a['robust_z']:.2f}, {a['direction']})")
        elif r.name == "generate_trend_plot":
            lines.append(f"**Graph** created for {', '.join(s['metric'] for s in res['series'])}.")
        elif r.name == "get_active_alerts":
            act = res["active_alerts"]
            lines.append(f"**Alerts**: {len(act)} active." + "".join(f"\n- [{a['severity']}] {a['message']}" for a in act[:5]))
        elif r.name == "evaluate_recommendation":
            lines.append(f"**Recommendation check**: {res.get('status', res.get('note'))} — {res.get('criterion', '')}. {res.get('caution', '')}")
        elif r.name == "explain_change":
            st = res["largest_step"]
            lines.append(f"**Largest step** in {res['metric']} at {st['at'][:19]}: {f(st['step'])}. Related metrics by correlation: " + ", ".join(f"{m['metric']} (r={f(m['pearson_r'])})" for m in res["related_metrics"][:4]) + ". Association is not causation.")
        elif r.name == "read_chart_table":
            if not res.get("available"):
                lines.append(f"- Chart model: {res.get('note')}")
            for rd in res.get("readings", []):
                cc = rd["cross_check_with_pixel_reader"]
                tbl = rd.get("table") or {}
                ser = "; ".join(f"{k}: last {v[-1]['value']:g} at {v[-1]['x']}" for k, v in (tbl.get("series") or {}).items() if v)
                lines.append(f"**chart-model reading** of {rd['chart_title'] or rd['chart_id']}: {ser or rd.get('error') or 'no table'} (cross-check with pixel reader: {cc['status']}).")
        elif r.name == "ask_chart_model":
            lines.append(f"**Chart-model answer** ({res.get('chart_title') or res.get('chart_id')}): {res.get('chart_model_answer')} — model estimate.")
        elif r.name == "list_metrics":
            lines.append("**Available metrics:** " + ", ".join(f"{m['metric']} ({m['status']}, {m['count']} pts)" for m in res["metrics"][:15]))
        elif r.name == "compare_data_sources":
            lines.append(f"**Source comparison** for {res.get('metric')}: {res.get('paired_samples', 0)} paired samples, agreement {f(res.get('agreement_pct'))} %, mean abs difference {f(res.get('mean_abs_difference'))}. {res.get('note', '')}")
        else:
            lines.append(f"- {r.name}: done.")
    return "\n".join(lines) if lines else "No computed results are available yet."
