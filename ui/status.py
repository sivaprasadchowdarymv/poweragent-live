"""Separate status indicators: capture · extraction · analysis · AI response."""

from __future__ import annotations

import streamlit as st

from core.runtime import Runtime

CSS = """
<style>
.lda-pills{display:flex;gap:.5rem;flex-wrap:wrap;margin:.1rem 0 .4rem}
.lda-pill{border:1px solid rgba(128,128,128,.35);border-radius:999px;padding:.15rem .7rem;font-size:.82rem;line-height:1.5}
.lda-dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;margin-right:.35rem;vertical-align:middle}
.lda-on{background:#0ca30c}.lda-busy{background:#2a78d6}.lda-warn{background:#fab219}.lda-off{background:#8a8985}.lda-err{background:#d03b3b}
</style>
"""


def _pill(label: str, state: str, cls: str, detail: str = "") -> str:
    d = f" · {detail}" if detail else ""
    return f'<span class="lda-pill"><span class="lda-dot {cls}"></span><b>{label}</b>: {state}{d}</span>'


@st.fragment(run_every=1.0)
def status_bar(rt: Runtime) -> None:
    cap = rt.capture
    cm = cap.metrics.summary()
    cstate = cap.state
    ccls = {"running": "lda-on", "paused": "lda-warn", "stopped": "lda-off"}[cstate]
    if cap.metrics.consecutive_errors:
        ccls, cstate = "lda-err", f"{cstate} (error)"
    cdetail = f"{cm['interval_mean_s']:.2f}s avg" if cm["interval_mean_s"] else f"target {cap.interval_s:g}s"
    ex = rt.extraction
    em = ex.metrics.summary()
    estate = ex.status if ex.alive or ex.status != "idle" else "stopped"
    ecls = {"extracting": "lda-busy", "idle": "lda-on", "stopped": "lda-off", "error": "lda-err"}.get(estate, "lda-off")
    edetail = f"OCR {em['ocr_ms']['mean']:.0f} ms" if em["ocr_ms"]["mean"] else ""
    if ex.chart_model_busy:
        edetail += " · chart-model reading"
    if ex.vision_busy:
        edetail += " · vision model busy"
    astate = rt.analysis_status
    acls = "lda-busy" if astate != "idle" else "lda-on"
    rstate, rcls = ("ready", "lda-on") if rt.router.enabled else ("no model", "lda-err")
    st.markdown(
        CSS
        + '<div class="lda-pills">'
        + _pill("Capture", cstate, ccls, cdetail)
        + _pill("Extraction", estate, ecls, edetail)
        + _pill("Analysis", astate, acls)
        + _pill("AI response", rstate, rcls, ", ".join(sorted({e.split(":")[0] for e in rt.settings.models_for("master")})))
        + "</div>",
        unsafe_allow_html=True,
    )
