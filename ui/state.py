"""Session/process state: one Runtime per server process, one conversation per browser session."""
from __future__ import annotations

from typing import List

import streamlit as st

from agents.references import ConversationState
from chat.conversations import Conversation, ConversationStore
from config import ConfigError, Settings, load_settings, setup_logging
from core.runtime import Runtime
from learning.feedback import FeedbackStore
from learning.preferences import PreferenceStore
from llm.model_router import ModelRouter, ProviderStatus
from vision.ocr import configure_tesseract


@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    """Reruns and new tabs reuse the same workers, so background threads are never duplicated."""
    settings = load_settings()
    setup_logging(settings.log_level)
    configure_tesseract(settings.tesseract_cmd or None)
    return Runtime(settings)


@st.cache_data(ttl=60, show_spinner=False)
def provider_status(_settings: Settings) -> List[ProviderStatus]:
    return ModelRouter(_settings).status()


def password_ok(rt: Runtime) -> bool:
    pw = rt.settings.app_password
    if not pw or st.session_state.get("auth_ok"):
        return True
    entered = st.text_input("Password", type="password")
    if entered and entered == pw:
        st.session_state.auth_ok = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    return False


def stores(rt: Runtime):
    return ConversationStore(rt.db), FeedbackStore(rt.settings.data_dir), PreferenceStore(rt.settings.data_dir)


def init_session(rt: Runtime) -> None:
    ss = st.session_state
    ss.setdefault("upload_cache", {})
    if "conv_id" not in ss:
        new_conversation(rt)


def new_conversation(rt: Runtime) -> None:
    conv = ConversationStore(rt.db).create()
    st.session_state.conv_id = conv.id
    st.session_state.conv_state = ConversationState(conv.id)


def open_conversation(conv_id: str) -> None:
    st.session_state.conv_id = conv_id
    st.session_state.conv_state = ConversationState(conv_id)


def current_conversation(rt: Runtime) -> Conversation:
    conv = ConversationStore(rt.db).get(st.session_state.conv_id)
    if conv is None:
        new_conversation(rt)
        conv = ConversationStore(rt.db).get(st.session_state.conv_id)
    assert conv is not None
    return conv


__all__ = ["ConfigError", "get_runtime", "provider_status", "init_session", "new_conversation", "current_conversation",
           "open_conversation", "stores", "password_ok"]
