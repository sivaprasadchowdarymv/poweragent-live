"""
Role-based routing across Groq (text), OpenRouter (images) and optional local Ollama
(unchanged design from RAG∞ Ultimate).

    router.chat("master", messages, tools=...)   text answer  -> groq:openai/gpt-oss-120b, ...
    router.chat("vision", messages, images=[..]) image answer -> openrouter:google/gemma-4-31b-it:free, ...

Each role has an ordered chain of "provider:model" entries (llm/catalog.py).
Entries whose provider has no API key are skipped. The router moves to the
next entry only for recoverable errors (rate limit, model down, no credits,
timeout, empty answer); auth errors, moderation refusals and malformed
requests stop immediately.

Every result records the provider and model that answered, the attempts made,
the tokens and (OpenRouter only) the real cost in USD.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import Settings, get_logger
from llm.catalog import PROVIDERS, ROLES, split
from llm.providers import FALLBACK_KINDS, ChatResult, KeyInfo, LLMError, make_provider

log = get_logger("router")

__all__ = ["ModelRouter", "RoutedResult", "ProviderStatus", "ROLES"]

# Hidden reasoning budget per role: small jobs and image reading need none.
_ROLE_REASONING = {"fast": "none", "verifier": "none", "vision": "none"}


@dataclass
class ProviderStatus:
    name: str
    label: str
    configured: bool
    connected: bool = False
    models: List[str] = field(default_factory=list)
    message: str = ""
    key: Optional[KeyInfo] = None
    used_for: str = ""


@dataclass
class RoutedResult:
    result: ChatResult
    attempts: List[Tuple[str, str, str]]  # (provider, model, outcome)
    latency_ms: float

    @property
    def cost(self) -> float:
        return self.result.cost


_COOLDOWN: Dict[Tuple[str, str], float] = {}  # (provider, model) -> monotonic time until which it is skipped
_DISCOVERED: Dict[str, List[str]] = {}


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.by_name = {name: make_provider(name, settings) for name in PROVIDERS}
        self.providers = [p for p in self.by_name.values() if p.configured()]

    def discovered_free_vision(self) -> List[str]:
        """Free (':free') image-capable Gemma / Qwen models currently listed by OpenRouter - appended
        to the vision chain so the app keeps working when model ids change."""
        p = self.by_name.get("openrouter")
        if p is None or not p.configured() or self.settings.model_profile == "local":
            return []
        if "ids" not in _DISCOVERED:
            try:
                ids = [str(m.get("id")) for m in p.raw_models()
                       if str(m.get("id", "")).endswith(":free")
                       and "image" in ((m.get("architecture") or {}).get("input_modalities") or [])
                       and any(f in str(m.get("id")) for f in ("gemma", "qwen"))]
                _DISCOVERED["ids"] = sorted(ids, key=lambda i: ("gemma" not in i, i))[:6]
            except LLMError:
                _DISCOVERED["ids"] = []
        return list(_DISCOVERED["ids"])

    # ------------------------------------------------------------------ models
    def chain(self, role: str):
        """[(provider, model)] for the role, keeping only providers that have a key."""
        out = []
        entries = list(self.settings.models_for(role))
        if role == "vision":
            entries += [f"openrouter:{m}" for m in self.discovered_free_vision() if f"openrouter:{m}" not in entries]
        for entry in entries:
            name, model = split(entry)
            p = self.by_name.get(name)
            if p is not None and p.configured():
                out.append((p, model))
        return out

    def has_role(self, role: str) -> bool:
        return bool(self.chain(role))

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    # -------------------------------------------------------------------- chat
    def chat(self, role: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", images: Optional[Sequence[bytes]] = None,
             max_tokens: Optional[int] = None, reasoning: Optional[str] = None,
             json_mode: bool = False, temperature: Optional[float] = None) -> RoutedResult:
        t0 = time.perf_counter()
        attempts: List[Tuple[str, str, str]] = []
        chain = self.chain(role)
        if not chain:
            need = "OPENROUTER_API_KEY (the screen is read by Gemma / Qwen)" if role == "vision" else \
                "GROQ_API_KEY (free at console.groq.com/keys)"
            raise LLMError(f"No AI model is configured for this task. Add {need} to .env or the app's secrets, "
                           "or use MODEL_PROFILE=local with Ollama.", "no_provider")
        effort = reasoning or _ROLE_REASONING.get(role) or self.settings.reasoning_effort
        last: Optional[LLMError] = None
        now = time.monotonic()
        live = [(p, m) for p, m in chain if _COOLDOWN.get((p.name, m), 0) <= now] or chain
        for provider, model in live:
            try:
                result = provider.chat(model, messages, tools=tools, tool_choice=tool_choice,
                                       temperature=self.settings.llm_temperature if temperature is None else temperature,
                                       max_tokens=max_tokens or self.settings.max_output_tokens, images=images,
                                       reasoning=effort, json_mode=json_mode)
                attempts.append((provider.name, model, "ok"))
                return RoutedResult(result, attempts, (time.perf_counter() - t0) * 1000)
            except LLMError as exc:
                attempts.append((provider.name, model, exc.kind))
                last = exc
                if exc.kind in ("rate_limit", "quota", "unavailable", "model"):
                    _COOLDOWN[(provider.name, model)] = time.monotonic() + (600 if exc.kind == "model" else 60)
                if exc.kind not in FALLBACK_KINDS:
                    raise
                log.warning("%s/%s failed (%s); trying the next model for '%s'", provider.name, model, exc.kind, role)
        assert last is not None
        last.user_message += " No other model in the chain could answer."
        last.attempts = attempts  # type: ignore[attr-defined]
        raise last

    # ------------------------------------------------------------------ status
    def status(self) -> List[ProviderStatus]:
        out = []
        for name, p in self.by_name.items():
            used = {"groq": "text answers", "openrouter": "reading the screen (images)", "ollama": "local models"}[name]
            st = ProviderStatus(p.name, p.label, p.configured(), used_for=used)
            if st.configured:
                try:
                    st.key = p.key_info()
                    st.connected = True
                except LLMError as exc:
                    st.message = exc.user_message
                if st.connected:
                    try:
                        st.models = p.list_models()
                    except LLMError:
                        st.models = []
            out.append(st)
        return out
