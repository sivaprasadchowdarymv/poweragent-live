"""Alerts, History, Settings and Diagnostics tabs."""

from __future__ import annotations

import os
import platform
import sys
import threading
import time

import pandas as pd
import streamlit as st
from PIL import Image

from capture.screen import Region, list_monitors
from capture.worker import Frame, fingerprint
from analysis import recommendations as recs
from analysis.alerts import AlertRule
from config import ConfigError
from core.models import utcnow
from core.reporting import build_report
from core.runtime import Runtime
from vision.extraction import analyze_frame
from vision.ocr import ocr_status
from llm.catalog import PROFILE_LABELS
from ui.state import new_conversation, provider_status

SEV_ICON = {"info": "ℹ️", "warning": "⚠️", "critical": "🛑"}


# ------------------------------------------------------------------ alerts
def alerts_tab(rt: Runtime) -> None:
    names = rt.service.metric_names()
    with st.form("rule_form", clear_on_submit=False):
        st.markdown("##### New alert rule")
        c = st.columns(4)
        metric = c[0].selectbox("Metric", names or ["(no metrics yet)"])
        kind = c[1].selectbox("Condition", ["above", "below", "outside", "anomaly", "stale"], help="'stale' fires when no new data arrives.")
        upper = c[2].number_input("Threshold / upper", value=0.0, format="%.6g")
        lower = c[3].number_input("Lower (for 'outside')", value=0.0, format="%.6g")
        d = st.columns(4)
        persist = d[0].number_input("Must persist (s)", 0.0, 86400.0, 10.0)
        cooldown = d[1].number_input("Cooldown after resolve (s)", 0.0, 86400.0, 120.0)
        stale = d[2].number_input("Stale after (s)", 1.0, 86400.0, 30.0)
        sev = d[3].selectbox("Severity", ["info", "warning", "critical"], index=1)
        if st.form_submit_button("Add rule", disabled=not names):
            try:
                rt.alerts.add_rule(AlertRule(metric, kind, upper if kind in ("above", "below", "outside") else None, lower if kind == "outside" else None, persist, sev, cooldown, stale))
                rt.save_rules()
                st.success("Rule added. Rules are evaluated after each screen extraction, or with the button below.")
            except ValueError as exc:
                st.error(str(exc))
    if rt.alerts.rules:
        for r in list(rt.alerts.rules.values()):
            row = st.container(horizontal=True, vertical_alignment="center")
            row.markdown(f"{SEV_ICON[r.severity]} **{r.severity}** · {r.describe()} · cooldown {r.cooldown_s:g}s")
            if row.button("Remove", key=f"rm_{r.id}", type="tertiary"):
                rt.alerts.remove_rule(r.id)
                rt.save_rules()
                st.rerun()
        if st.button("Evaluate rules now"):
            ev = rt.alerts.evaluate()
            st.write(ev or "No state changes.")
    active = rt.db.alerts(active_only=True)
    st.markdown(f"##### Active alerts ({len(active)})")
    for a in active:
        ev = a["evidence"]
        st.markdown(f"{SEV_ICON.get(a['severity'], '')} **{a['severity'].upper()}** — {a['message']} · since {a['opened_at'][:19]} · seen {a['occurrences']}× · latest {ev.get('latest', '—')} {ev.get('unit', '')} ({ev.get('status', '')})")
    hist = rt.db.alerts(limit=100)
    if hist:
        st.markdown("##### Alert history")
        st.dataframe(pd.DataFrame(hist)[["severity", "message", "opened_at", "last_seen_at", "resolved_at", "occurrences"]], hide_index=True, height=240)


# ------------------------------------------------------------------ history
def history_tab(rt: Runtime) -> None:
    cat = rt.service.metric_catalog()
    st.markdown(f"##### Stored measurements · {rt.db.count_observations():,} observations")
    if cat:
        st.dataframe(pd.DataFrame(cat).drop(columns=["metric_key"]), hide_index=True, height=220)
    c = st.container(horizontal=True)
    if c.button("Build analysis report"):
        st.session_state.report_md = build_report(rt.service, rt.alerts, "all")
    if st.session_state.get("report_md"):
        c.download_button("⬇ Download report (.md)", st.session_state.report_md, "poweragent_report.md", "text/markdown")
    obs = rt.db.query_observations(limit=200_000)
    if not obs.empty:
        c.download_button("⬇ Export observations (.csv)", obs.drop(columns=["ts"]).assign(meta=obs["meta"].astype(str)).to_csv(index=False), "observations.csv", "text/csv")

    st.markdown("##### Recommendations")
    items = rt.db.recommendations(limit=30)
    if not items:
        st.caption("Recommendations appear here when you ask what to do next.")
    for r in items:
        with st.expander(f"{r['created_at'][:19]} · {r['metric']} · {r['status']}"):
            st.markdown(r["text"][:1500])
            st.caption("Success criterion: " + recs.describe_criteria(r["criteria"]))
            if st.button("Evaluate now", key=f"ev_{r['id']}"):
                res = recs.evaluate(rt.db, r)
                rt.db.set_recommendation_outcome(r["id"], res["status"], res)
                st.json(res, expanded=False)
            elif r.get("outcome"):
                st.json(r["outcome"], expanded=False)

    st.markdown("##### Screen extraction timeline")
    if rt.service.screen_history:
        tl = [{"captured_at": h["captured_at"][:19], "frame": h["frame_seq"], "charts": len(h["charts"]), "kpis": len(h["kpis"]), "series": sum(len(c["series"]) for c in h["charts"])} for h in list(rt.service.screen_history)[-50:]]
        st.dataframe(pd.DataFrame(tl[::-1]), hide_index=True, height=200)
    else:
        st.caption("No screen extractions yet.")
    frames = rt.stored_frames()
    if frames:
        st.markdown(f"##### Replay stored captures ({len(frames)})")
        i = st.slider("Frame", 0, len(frames) - 1, len(frames) - 1)
        st.image(Image.open(frames[i]), caption=frames[i].name, width="stretch")
        if st.button("Re-analyze this frame"):
            img = Image.open(frames[i]).convert("RGB")
            fr = Frame(-1, utcnow(), time.monotonic(), img, Region(0, 0, img.width, img.height), "replay", fingerprint(img))
            st.json(analyze_frame(fr).summary(), expanded=False)

    st.markdown("##### Conversations")
    for conv in rt.db.conversations(limit=15):
        with st.expander(f"{conv['created_at'][:19]} · {conv['messages']} messages"):
            for m in rt.db.messages(conv["id"])[-20:]:
                st.markdown(f"**{m['role']}:** {m['content'][:600]}")

    st.markdown("##### Retention & deletion")
    st.caption(f"Screen observations older than {rt.settings.retention_days} days are deleted automatically; file data keeps its own timestamps. Starting a new conversation never deletes measurements.")
    d = st.columns(4)
    if d[0].button("New conversation"):
        new_conversation(rt)
        st.rerun()
    confirm = d[3].checkbox("I understand deletion is permanent", key="del_confirm")
    if d[1].button("Delete chat history", disabled=not confirm):
        rt.db.delete_conversations()
        new_conversation(rt)
        st.success("Chat history deleted.")
    if d[2].button("Delete all measurements", disabled=not confirm):
        n = rt.db.delete_all_measurements()
        rt.service.datasets.clear()
        rt.service.latest_screen = None
        rt.service.screen_history.clear()
        k = rt.delete_stored_frames()
        st.success(f"Deleted {n:,} observations and {k} stored frame(s).")


# ------------------------------------------------------------------ settings
def settings_tab(rt: Runtime) -> None:
    s = rt.settings
    with st.form("settings_form"):
        st.markdown("##### AI models (same models as RAG∞ Ultimate)")
        c = st.columns(3)
        profiles = list(PROFILE_LABELS)
        profile = c[0].selectbox("Model profile", profiles, index=profiles.index(s.model_profile), format_func=PROFILE_LABELS.get)
        img = c[1].toggle("Show the screen to the vision model for screen questions", s.image_answers, help="Sends chart crops of the latest capture to Gemma/Qwen (OpenRouter) or local Ollama, only for questions about what the screen shows.")
        local_only = c[2].toggle("Local-only mode", s.local_only, help="Only Ollama on this PC is used; nothing leaves the computer.")
        d = st.columns(3)
        verify = d[0].toggle("Extra LLM fact-check (Llama 3.3 70B)", s.verify_with_llm)
        mode = d[1].selectbox("Answer mode", ["agent", "quick"], index=0 if s.agent_mode == "agent" else 1, help="agent: the model may call analysis tools for why/compare/advice questions")
        ollama_on = d[2].toggle("Add local Ollama as last fallback", s.ollama_enabled)
        st.markdown("##### Capture & extraction")
        e = st.columns(4)
        cap = e[0].number_input("Capture interval (s)", 0.25, 600.0, float(s.capture_interval_s))
        ext = e[1].number_input("Min. extraction interval (s)", 0.5, 600.0, float(s.extraction_interval_s))
        ocr_on = e[2].toggle("OCR + chart reading", s.ocr_enabled)
        vis_on = e[3].toggle("Background AI reading (vision-first)", s.vision_enabled, help="The vision model reads changed frames every interval and stores its inferences over time.")
        f = st.columns(4)
        vis_int = f[0].number_input("AI reading interval (s)", 10.0, 86400.0, float(s.vision_interval_s), help="Free models allow a limited number of requests per day; 60 s or more is recommended.")
        dedup = f[1].number_input("Duplicate threshold", 0.0, 1.0, float(s.dedup_threshold), format="%.3f")
        store = f[2].toggle("Keep screenshots for replay (opt-in)", s.store_screenshots)
        ret = f[3].number_input("Retention (days)", 1, 3650, int(s.retention_days))
        st.markdown("##### Chart model (optional local server: Granite Vision / ChartAst)")
        g = st.columns(4)
        ca_on = g[0].toggle("Use chart model", s.chart_model_enabled)
        ca_url = g[1].text_input("Chart model server URL", s.chart_model_url)
        ca_int = g[2].number_input("Chart model interval (s)", 2.0, 86400.0, float(s.chart_model_interval_s))
        ca_max = g[3].number_input("Charts per reading", 1, 10, int(s.chart_model_max_charts))
        ca_to = st.number_input("Chart model timeout (s)", 5.0, 3600.0, float(s.chart_model_timeout_s))
        if st.form_submit_button("Apply settings", type="primary"):
            try:
                new = s.with_updates(
                    model_profile=profile, image_answers=img, local_only=local_only, verify_with_llm=verify, agent_mode=mode, ollama_enabled=ollama_on,
                    capture_interval_s=cap, extraction_interval_s=ext, ocr_enabled=ocr_on, vision_enabled=vis_on, vision_interval_s=vis_int, dedup_threshold=dedup,
                    store_screenshots=store, retention_days=int(ret),
                    chart_model_enabled=ca_on, chart_model_url=ca_url, chart_model_interval_s=ca_int, chart_model_max_charts=int(ca_max), chart_model_timeout_s=ca_to,
                )
                rt.apply_settings(new)
                provider_status.clear()
                st.success("Settings applied.")
            except ConfigError as exc:
                st.error(str(exc))
    st.caption("Chains in use: " + " · ".join(f"**{r}**: {', '.join(rt.settings.models_for(r)) or '-'}" for r in ("master", "fast", "verifier", "vision")))
    if st.button("Test chart model server"):
        h = rt.chart_model.health()
        if h.get("ok"):
            st.success(f"Chart model server is up: {h.get('model')} on {h.get('device')} · {h.get('dtype') or h.get('quant')} · GPU memory {h.get('gpu_memory_gb')} GB · output {h.get('output_format')} · busy {h.get('busy')}")
        else:
            st.error(f"{h.get('error')} — start it with: python chart_model_server\\granite_server.py (see chart_model_server/README.md)")


# ------------------------------------------------------------------ diagnostics
def diagnostics_tab(rt: Runtime) -> None:
    c = st.columns(3)
    ok, msg = ocr_status()
    c[0].metric("AI providers", sum(p.connected for p in provider_status(rt.settings)))
    c[1].metric("OCR", "available" if ok else "missing", msg[:40])
    try:
        c[2].metric("Monitors", len(list_monitors()))
    except Exception:
        c[2].metric("Monitors", 0, "no display")
    if not ok:
        st.warning(msg)
    st.dataframe(pd.DataFrame([{"provider": p.label, "configured": p.configured, "connected": p.connected, "used for": p.used_for,
                                "models": len(p.models), "message": p.message} for p in provider_status(rt.settings)]), hide_index=True)
    st.markdown("##### Vision test (does the AI see your screen?)")
    if st.button("🖼️ Send the latest frame to the vision model"):
        frame = rt.latest_frame()
        if frame is None:
            st.warning("No frame yet: click 'Share a tab' in the sidebar (or start capture) first.")
        else:
            from llm.providers import LLMError
            from vision.vision_model import to_jpeg

            with st.spinner("Asking the vision model…"):
                try:
                    out = rt.router.chat("vision", [{"role": "user", "content": "Describe this dashboard: chart titles, "
                                         "axes, series, the trend of each line, visible values and anything unusual."}],
                                         images=[to_jpeg(frame.image, rt.settings.image_max_px)], max_tokens=700)
                    st.success(f"Answered by {out.result.provider}: {out.result.model} · attempts: {out.attempts}")
                    st.markdown(out.result.content)
                except LLMError as exc:
                    st.error(f"{exc.user_message} · attempts: {getattr(exc, 'attempts', [])}")
        a = rt.service.latest_screen
        if a is not None:
            st.caption(f"Pixel chart reader on the latest frame: {len(a.charts)} chart(s), {len(a.kpis)} KPI(s), "
                       f"{len(a.text_lines)} OCR text line(s). Chains: vision = {', '.join(rt.settings.models_for('vision')) or 'none'}")
    st.markdown("##### Performance")
    cm = rt.capture.metrics.summary()
    em = rt.extraction.metrics.summary()
    perf = {
        "capture state": rt.capture.state,
        "target interval (s)": rt.capture.interval_s,
        "measured interval mean / p95 / max (s)": f"{cm['interval_mean_s'] or 0:.3f} / {cm['interval_p95_s'] or 0:.3f} / {cm['interval_max_s'] or 0:.3f}",
        "grab time mean / p95 (ms)": f"{cm['grab_mean_ms'] or 0:.1f} / {cm['grab_p95_ms'] or 0:.1f}",
        "captures / enqueued / duplicates / dropped": f"{cm['captures']} / {cm['enqueued']} / {cm['duplicates_skipped']} / {cm['dropped_frames']}",
        "frame queue depth (bounded)": f"{rt.capture.queue_depth()} / {rt.settings.frame_queue_size}",
        "extraction processed / skipped frames": f"{em['processed_frames']} / {em['skipped_frames']}",
        "OCR mean / p95 (ms)": f"{em['ocr_ms']['mean'] or 0:.0f} / {em['ocr_ms']['p95'] or 0:.0f}",
        "chart reading mean (ms)": f"{em['chart_ms']['mean'] or 0:.0f}",
        "capture→extraction delay mean (ms)": f"{em['capture_to_extraction_delay_ms']['mean'] or 0:.0f}",
        "effective extraction interval (s)": f"{em['effective_interval_s']:.2f}",
        "vision calls / busy-skips / errors": f"{em['vision_calls']} / {em['vision_skipped_busy']} / {em['vision_errors']}",
        "vision latency mean (ms)": f"{em['vision_ms']['mean'] or 0:.0f}",
        "Chart model enabled / URL": f"{rt.settings.chart_model_enabled} / {rt.settings.chart_model_url}",
        "Chart model calls / busy-skips / errors": f"{em['chart_model_calls']} / {em['chart_model_skipped_busy']} / {em['chart_model_errors']}",
        "Chart model latency mean (ms)": f"{em['chart_model_ms']['mean'] or 0:.0f}",
        "capture threads started (should stay 1 per start)": rt.capture.threads_started,
        "extraction threads started": rt.extraction.threads_started,
        "live threads": ", ".join(sorted(t.name for t in threading.enumerate() if t.name.startswith("lda-"))) or "none",
        "process memory (MB)": _rss_mb(),
        "stored observations": rt.db.count_observations(),
        "database": str(rt.settings.db_path),
    }
    st.dataframe(pd.DataFrame({"value": {k: str(v) for k, v in perf.items()}}), height=560)
    if em["chart_model_last_error"]:
        st.warning(f"Last chart-model error: {em['chart_model_last_error']}")
    if cm["last_error"] or em["last_error"] or em["vision_last_error"]:
        st.warning(f"Last errors — capture: {cm['last_error'] or 'none'} · extraction: {em['last_error'] or 'none'} · vision: {em['vision_last_error'] or 'none'}")
    benchmark_panel(rt)
    st.caption(f"Python {sys.version.split()[0]} · {platform.platform()} · PID {os.getpid()}")


def benchmark_panel(rt: Runtime) -> None:
    import json

    from metrics.priors import latest_run

    st.markdown("##### ChartNet accuracy benchmark")
    st.caption(
        "Scores the chart readers against IBM ChartNet ground-truth tables (free CDLA-Permissive-2.0 subset; streamed, only the sampled rows are downloaded). "
        "The measured per-chart-type accuracy then scales the confidence of live readings. Needs `pip install datasets` and internet."
    )
    c = st.columns([1, 1, 2])
    n = c[0].number_input("Charts", 10, 2000, 100, step=10, key="bench_n")
    readers = c[1].multiselect("Readers", ["pixel", "model"], default=["pixel"] + (["model"] if rt.settings.chart_model_enabled else []), key="bench_readers")
    if c[2].button("Run ChartNet benchmark", disabled=not readers):
        st.toast(rt.start_benchmark(int(n), ",".join(readers)))
    stt = rt.benchmark_status()
    if stt["state"] != "idle":
        st.caption(f"Benchmark: {stt['state']}")
        if stt["log"]:
            st.code(stt["log"][-1200:], language=None)
        if stt["state"] == "finished" and st.button("Use these results for confidence"):
            pri = rt.reload_priors()
            st.success(f"Loaded priors from run {pri.get('_run')}.")
    run = latest_run(rt.settings.data_dir)
    if run is None:
        st.caption("No benchmark run yet. Live confidences use only each reader's own checks.")
        return
    data = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = [{"reader": r, **a} for r, a in data["overall"].items()]
    st.markdown(f"Latest run **{run.name}** · {data['meta']['samples']} charts · subset `{data['meta']['subset']}`" + (f" · model `{data['meta']['model']}`" if data['meta'].get("model") else ""))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    types = data["by_chart_type"]
    table = pd.DataFrame({r: {t: v["f1@5"] for t, v in d.items()} for r, d in types.items()})
    if not table.empty:
        st.caption("F1@5% by chart type")
        st.dataframe(table.style.format("{:.0%}"), height=min(420, 38 + 35 * len(table)))
    rep = run / "report.md"
    if rep.exists():
        st.download_button("⬇ Benchmark report (.md)", rep.read_text(encoding="utf-8"), f"chartnet_{run.name}.md", "text/markdown")
    st.caption("Applied to live confidence: " + ("yes (run " + str(rt.service.priors.get("_run")) + ")" if rt.service.priors else "not yet — press 'Use these results' after a run, or restart the app"))


def _rss_mb() -> str:
    try:
        import psutil  # type: ignore[import-untyped]

        return f"{psutil.Process().memory_info().rss / 1e6:.0f}"
    except ImportError:
        try:
            import resource

            return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} (peak)"
        except ImportError:
            return "n/a"
