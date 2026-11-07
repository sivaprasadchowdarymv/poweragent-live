"""Browser screen capture for the hosted (Streamlit Cloud) app.

On a server, `mss` would capture the SERVER's screen, which is useless. Instead the visitor's
browser shares a tab / window / screen (getDisplayMedia) and sends a downscaled JPEG every few
seconds; the server runs the same OCR + chart-reader pipeline on it. Nothing is captured unless
the visitor clicks "Share" and picks what to share; the browser shows a sharing indicator.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from core.runtime import Runtime

_component = components.declare_component("browser_capture", path=str(Path(__file__).parent / "browser_capture"))


def browser_capture_panel(rt: Runtime) -> None:
    """Rendered in the sidebar so sharing keeps running while you chat on other pages."""
    st.markdown("**🌐 Live screen share**")
    interval = float(st.session_state.get("bc_interval", max(2.0, rt.settings.capture_interval_s)))
    value = _component(interval=interval, max_px=1600, quality=0.8, key="browser_capture", default=None)
    if not value or value.get("seq") == st.session_state.get("bc_last_seq"):
        return
    st.session_state.bc_last_seq = value.get("seq")
    try:
        raw = base64.b64decode(str(value["jpeg"]).split(",", 1)[1])
        if len(raw) > 8_000_000:
            st.warning("Frame too large; skipped.")
            return
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not decode the shared frame: {exc}")
        return
    a = rt.ingest_browser_frame(img)
    if a is not None:
        st.session_state.bc_info = f"Frame {value['seq']}: {len(a.charts)} chart(s), {len(a.kpis)} KPI(s) read."
    st.caption(st.session_state.get("bc_info", ""))
