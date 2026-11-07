"""
Model catalog: which model plays which role (same models and rule as RAG∞ Ultimate).

The rule of this app: **text goes to Groq (free, fast), images go to Gemma / Qwen (OpenRouter).**

    master     writes the cited answer about live data       -> Groq gpt-oss-120b
    fast       tiny jobs                                      -> Groq gpt-oss-20b
    verifier   optional LLM cross-check (off by default)      -> Groq Llama 3.3 70B (other family)
    vision     anything that must SEE the screen:             -> OpenRouter Gemma 4 / Qwen 3.8
               * questions about what is on screen / a chart (chart crops are attached)
               * optional background notes on captured frames (VISION_NOTES, off by default)

Every chain entry is "provider:model". Entries whose provider is not configured are
skipped; the router walks the chain in order on recoverable errors (rate limit,
model down, no credits, timeout). The last text fallback is free Gemma on OpenRouter.

Profiles (MODEL_PROFILE):
    free     Groq free tier for text + Gemma 4 ":free" for images      -> $0
    quality  Groq for text + Qwen 3.8 (paid, cents) first for images    -> better chart reading
    local    Ollama on this computer for everything                    -> $0, nothing leaves the PC

Override any chain with MASTER_MODELS / FAST_MODELS / VERIFIER_MODELS / VISION_MODELS,
e.g. VISION_MODELS=openrouter:qwen/qwen3.8-flash,openrouter:google/gemma-4-31b-it:free
"""
from __future__ import annotations

from typing import Dict, Tuple

ROLES = ("master", "fast", "verifier", "vision")
PROVIDERS = ("groq", "openrouter", "ollama")

# Groq production models (console.groq.com/docs/models) — same chain as RAG∞ Ultimate
GROQ_120B = "groq:openai/gpt-oss-120b"
GROQ_20B = "groq:openai/gpt-oss-20b"
GROQ_LLAMA70 = "groq:llama-3.3-70b-versatile"
GROQ_LLAMA8 = "groq:llama-3.1-8b-instant"
# OpenRouter multimodal models — same chain as RAG∞ Ultimate
GEMMA_31B_FREE = "openrouter:google/gemma-4-31b-it:free"
GEMMA_26B_FREE = "openrouter:google/gemma-4-26b-a4b-it:free"
GEMMA_31B = "openrouter:google/gemma-4-31b-it"
QWEN_FLASH = "openrouter:qwen/qwen3.8-flash"
QWEN_27B = "openrouter:qwen/qwen3.8-27b"
QWEN_VL72_FREE = "openrouter:qwen/qwen2.5-vl-72b-instruct:free"
QWEN_VL32_FREE = "openrouter:qwen/qwen2.5-vl-32b-instruct:free"
# Local (Ollama, OpenAI-compatible endpoint) — pull with `ollama pull <name>`
OLLAMA_TEXT = "ollama:qwen2.5:7b"
OLLAMA_TEXT_SMALL = "ollama:llama3.2:3b"
OLLAMA_VISION = "ollama:qwen2.5vl:7b"

_TEXT = {
    "master": (GROQ_120B, GROQ_LLAMA70, GEMMA_31B_FREE),
    "fast": (GROQ_20B, GROQ_LLAMA8),
    "verifier": (GROQ_LLAMA70, GROQ_20B),
}
PROFILES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "free": {**_TEXT, "vision": (GEMMA_31B_FREE, QWEN_VL72_FREE, GEMMA_26B_FREE, QWEN_VL32_FREE)},
    "quality": {**_TEXT, "vision": (QWEN_FLASH, GEMMA_31B, GEMMA_31B_FREE, QWEN_VL72_FREE)},
    "local": {"master": (OLLAMA_TEXT, OLLAMA_TEXT_SMALL), "fast": (OLLAMA_TEXT_SMALL, OLLAMA_TEXT),
              "verifier": (OLLAMA_TEXT,), "vision": (OLLAMA_VISION,)},
}
PROFILE_LABELS = {
    "free": "🆓 Free - Groq for text, free Gemma 4 / Qwen VL for images ($0)",
    "quality": "🖼️ Quality images - Groq for text, Qwen 3.8 for images (cents)",
    "local": "🔒 Local - Ollama on this PC for everything ($0, offline)",
}
DEFAULT_PROFILE = "free"


def chain_for(profile: str, role: str) -> Tuple[str, ...]:
    return PROFILES.get(profile, PROFILES[DEFAULT_PROFILE]).get(role, ())


def split(entry: str) -> Tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'); 'ollama:qwen2.5:7b' -> ('ollama', 'qwen2.5:7b').
    Unprefixed ids default to OpenRouter."""
    head, sep, rest = entry.partition(":")
    if sep and head in PROVIDERS:
        return head, rest
    return "openrouter", entry


def is_free(entry: str) -> bool:
    provider, model = split(entry)
    return provider in ("groq", "ollama") or model.endswith(":free")


def short_name(entry: str) -> str:
    """'groq:openai/gpt-oss-120b' -> 'gpt-oss-120b' (compact UI labels)."""
    return split(entry)[1].split("/", 1)[-1]
