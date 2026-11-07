"""PowerAgent Live Ultimate — Streamlit entry point.

    streamlit run app.py          (binds to 127.0.0.1:8501, see .streamlit/config.toml)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # optional .env for local runs
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

import streamlit as st  # noqa: E402

from config import ConfigError  # noqa: E402
from ui.analytics import analytics_tab  # noqa: E402
from ui.browser_capture import browser_capture_panel  # noqa: E402
from ui.chat import chat_page  # noqa: E402
from ui.data_sources import data_sources_tab  # noqa: E402
from ui.monitor import monitor_tab  # noqa: E402
from ui.panels import alerts_tab, diagnostics_tab, history_tab, settings_tab  # noqa: E402
from ui.state import get_runtime, init_session, new_conversation, open_conversation, password_ok, provider_status  # noqa: E402
from ui.status import status_bar  # noqa: E402
from ui.styles import CSS  # noqa: E402

st.set_page_config(page_title="PowerAgent Live", page_icon="⚡", layout="wide")


def sidebar(rt) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">⚡</div><div><div class="brand-title">PowerAgent Live</div>'
                    '<div class="brand-sub">Live dashboard intelligence</div></div></div>', unsafe_allow_html=True)
        if st.button("➕ New chat", width="stretch"):
            new_conversation(rt)
            st.rerun()
        for c in rt.db.conversations(limit=8):
            if c["messages"] and st.button(f"💬 {(c['title'] or 'Chat')[:32]}", key=f"c{c['id']}", width="stretch"):
                open_conversation(c["id"])
                st.rerun()
        st.divider()
        browser_capture_panel(rt)
        st.divider()
        for p in provider_status(rt.settings):
            icon = "🟢" if p.connected else ("🔴" if p.configured else "⚪")
            st.caption(f"{icon} {p.label} — {p.used_for}" + ("" if p.configured else " (not set)"))
        st.caption(f"Profile: **{rt.settings.model_profile}** · capture: {rt.capture.state}")


def main() -> None:
    try:
        rt = get_runtime()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    st.markdown(CSS, unsafe_allow_html=True)
    if not password_ok(rt):
        st.stop()
    init_session(rt)
    sidebar(rt)
    status_bar(rt)

    def page(fn, title, icon, default=False):
        return st.Page(lambda: fn(rt), title=title, icon=icon, url_path=title.lower().replace(" ", "_"), default=default)

    nav = st.navigation([
        page(chat_page, "Chat", "💬", True), page(monitor_tab, "Live Monitor", "🖥️"), page(data_sources_tab, "Data", "📁"),
        page(analytics_tab, "Analytics", "📈"), page(alerts_tab, "Alerts", "🚨"), page(history_tab, "History", "🗂️"),
        page(settings_tab, "Settings", "⚙️"), page(diagnostics_tab, "Diagnostics", "🩺"),
    ])
    nav.run()


main()
