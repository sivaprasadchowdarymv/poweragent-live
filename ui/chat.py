"""Chat page: the conversational interface (RAG∞ Ultimate look and flow)."""
from __future__ import annotations

import html

import plotly.io as pio
import streamlit as st

from agents.pipeline import answer_live
from agents.result import AgentStep, QueryResult
from core.runtime import Runtime
from learning.feedback import REASONS
from ui.state import current_conversation, stores

EXAMPLES = ["What is on my screen?", "Which metric is changing fastest?", "Compare the last 10 minutes with the previous 10 minutes",
            "Are there any anomalies?", "What should I do about the active alerts?", "Show me a graph of all metrics"]


def _pill(r: QueryResult) -> str:
    v = r.verification or {}
    txt, cls = {"verified": ("✓ Verified", "ok"), "partial": ("≈ Partly verified", "warn"),
                "unverified": ("⚠ Check the numbers", "bad")}.get(v.get("verdict", ""), ("", ""))
    if r.kind == "offline":
        txt, cls = "Computed results (no AI)", "info"
    if not txt:
        return ""
    if r.regenerated:
        txt += " · corrected once"
    return f'<span class="badge {cls}">{html.escape(txt)}</span>'


def _meta(r: QueryResult) -> str:
    bits = []
    if r.cached:
        bits.append("cache")
    if r.provider:
        bits.append(f"{r.provider}: {r.model}")
    if r.route == "vision":
        bits.append(f"🖼 {r.images_sent} screen image(s)")
    if r.tokens:
        bits.append(f"{r.tokens:,} tokens")
    if r.cost:
        bits.append(f"${r.cost:.4f}")
    bits.append(f"{r.latency:.1f} s")
    return '<span class="meta">' + html.escape(" · ".join(bits)) + "</span>"


def _step_text(s: AgentStep) -> str:
    if s.kind == "action":
        return f"🔧 `{s.tool}` {', '.join(f'{k}={v}' for k, v in s.args.items())}"
    if s.kind == "observation":
        return ""
    return f"• {s.text}"


def render_result(rt: Runtime, r: QueryResult, key: str) -> None:
    st.markdown(r.answer)
    head = _pill(r) + " " + _meta(r)
    if r.sources:
        head += " " + " ".join(f'<span class="chip">{html.escape(s["label"])}</span>' for s in r.sources[:8])
    st.markdown(head, unsafe_allow_html=True)
    for fid in r.figure_ids:
        fig = rt.service.figures.get(fid)
        if fig:
            st.plotly_chart(pio.from_json(__import__("json").dumps(fig["figure"])), key=f"{key}-{fid}", width="stretch")
    for rid in r.report_ids:
        md = rt.service.reports.get(rid)
        if md:
            st.download_button("⬇ Download report (.md)", md, f"{rid}.md", "text/markdown", key=f"{key}-{rid}")
    for w in r.warnings:
        st.caption(f"⚠️ {w}")
    with st.expander("Evidence and agent activity"):
        for s in r.steps:
            t = _step_text(s)
            if t:
                st.markdown(t)
        for e in r.evidence:
            st.markdown(f"**{e['label']}**")
            st.code(e["text"][:1500] or "(image)", language=None)


def _feedback(rt: Runtime, conv, msg, question: str) -> None:
    conv_store, fb, prefs = stores(rt)
    if not rt.settings.enable_feedback or msg.meta.get("feedback"):
        return
    c = st.columns([1, 1, 8])
    if c[0].button("👍", key=f"up{msg.id}"):
        fb.record(conversation_id=conv.id, message_id=str(msg.id), question=question, answer=msg.content, rating=1)
        conv_store.set_feedback(msg, {"rating": 1})
        st.rerun()
    if c[1].button("👎", key=f"dn{msg.id}"):
        st.session_state[f"why{msg.id}"] = True
    if st.session_state.get(f"why{msg.id}"):
        reasons = st.multiselect("What was wrong?", REASONS, key=f"rs{msg.id}")
        if st.button("Send feedback", key=f"sf{msg.id}"):
            fb.record(conversation_id=conv.id, message_id=str(msg.id), question=question, answer=msg.content, rating=-1, reasons=reasons)
            conv_store.set_feedback(msg, {"rating": -1, "reasons": reasons})
            changes = prefs.apply_feedback(-1, reasons)
            if changes:
                st.toast("Preferences adapted: " + "; ".join(changes))
            st.rerun()


def chat_page(rt: Runtime) -> None:
    conv = current_conversation(rt)
    conv_store, _, prefs = stores(rt)
    from ui.guide import QUESTIONS, checklist

    checklist(rt, compact=bool(conv.messages))
    if not conv.messages:
        st.markdown("#### Ask about your shared screen")
        for group, qs in QUESTIONS.items():
            st.caption(group)
            cols = st.columns(len(qs))
            for i, ex in enumerate(qs):
                if cols[i].button(ex, key=f"ex-{group}-{i}", width="stretch"):
                    st.session_state.pending_q = ex
                    st.rerun()
    last_q = ""
    for m in conv.messages:
        with st.chat_message(m.role):
            if m.role == "user":
                st.markdown(m.content)
                last_q = m.content
            else:
                r = QueryResult.from_dict(m.result or {"answer": m.content})
                render_result(rt, r, f"m{m.id}")
                _feedback(rt, conv, m, last_q)
    msg = st.chat_input("Ask about the screen or a chart - or attach an image (📎)…", accept_file=True,
                        file_type=["png", "jpg", "jpeg", "webp", "bmp"])
    q, files = None, []
    if msg is not None:
        q, files = (msg.text or "").strip(), list(msg.files or [])
    q = q or st.session_state.pop("pending_q", None) or ("Analyse this image and give your inferences." if files else None)
    if not q:
        return
    for f in files[:1]:
        from PIL import Image

        from llm.providers import LLMError

        with st.spinner(f"Reading {f.name} with OCR, the chart reader and the vision model…"):
            try:
                rt.analyze_image_now(Image.open(f), f"upload: {f.name}")
            except LLMError as exc:
                st.warning(f"The vision model could not pre-read the image ({exc.user_message}); trying again while answering.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not open {f.name}: {exc}")
        q = f"[image: {f.name}] {q}"
    prev = conv.last_question()
    memory = conv.memory(rt.settings.memory_turns)
    conv_store.add_user(conv, q)
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.status("Analyzing…", expanded=False) as box:
            def on_step(s: AgentStep) -> None:
                t = _step_text(s)
                if t:
                    box.write(t)
            with rt.chat_lock:
                result = answer_live(q, rt, rt.router, st.session_state.conv_state, rt.settings, memory, prev,
                                     prefs.load().to_prompt(), on_step, session_key=st.session_state.conv_id)
            box.update(label=f"Done in {result.latency:.1f} s", state="complete")
        conv_store.add_assistant(conv, result.answer, result.as_dict())
    st.rerun()
