"""
Central configuration for PowerAgent Live.

Every value can come from an environment variable, a `.env` file, or
Streamlit secrets (same scheme as RAG∞ Ultimate). Secrets (GROQ_API_KEY,
OPENROUTER_API_KEY, APP_PASSWORD) are never displayed or logged.

Routing rule (llm/catalog.py): text -> Groq (free), images -> Gemma 4 / Qwen 3.8
on OpenRouter, everything else (capture, OCR, chart reading, statistics,
verification) runs locally at 0 tokens. MODEL_PROFILE=local uses Ollama only.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import urlparse

try:  # optional: read a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_TITLE = "PowerAgent Live"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


class ConfigError(ValueError):
    """An invalid setting; the message is safe to show in the UI."""


# ---------------------------------------------------------------------------
# Lookup helpers: explicit mapping (tests) > environment > Streamlit secrets
# ---------------------------------------------------------------------------
def _secret(name: str) -> Optional[str]:
    try:
        import streamlit as st

        value = st.secrets.get(name)  # raises if no secrets file exists
    except Exception:
        return None
    return None if value is None else str(value)


class _Source:
    def __init__(self, env: Optional[Mapping[str, str]]):
        self.env = env

    def lookup(self, name: str) -> Optional[str]:
        if self.env is not None:
            v = self.env.get(name)
            return v if v is not None and str(v).strip() else None
        value = os.getenv(name)
        if value is not None and value.strip():
            return value
        return _secret(name)

    def str_(self, name: str, default: str) -> str:
        return (self.lookup(name) or "").strip() or default

    def int_(self, name: str, default: int, lo: int, hi: int) -> int:
        raw = self.lookup(name)
        if raw is None:
            return default
        try:
            return max(lo, min(hi, int(float(raw))))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be a whole number (got {raw!r}).") from exc

    def float_(self, name: str, default: float, lo: float, hi: float) -> float:
        raw = self.lookup(name)
        if raw is None:
            return default
        try:
            return max(lo, min(hi, float(raw)))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be a number (got {raw!r}).") from exc

    def bool_(self, name: str, default: bool) -> bool:
        value = self.lookup(name)
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def choice(self, name: str, default: str, allowed: Tuple[str, ...]) -> str:
        value = self.str_(name, default).lower()
        return value if value in allowed else default

    def models(self, name: str) -> Tuple[str, ...]:
        return tuple(m.strip() for m in self.str_(name, "").split(",") if m.strip())


@dataclass(frozen=True)
class Settings:
    # --- Providers: Groq for text, OpenRouter (Gemma / Qwen) for images, Ollama optional --------
    groq_api_key: str = field(default="", repr=False)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_enabled: bool = False          # also use local Ollama models (always on in the 'local' profile)
    local_only: bool = False              # never call cloud providers (Groq / OpenRouter)
    app_url: str = "https://github.com/poweragent-live"
    app_title: str = APP_TITLE
    model_profile: str = "free"           # free | quality | local   (llm/catalog.py)
    master_models: Tuple[str, ...] = ()   # empty = profile default
    fast_models: Tuple[str, ...] = ()
    verifier_models: Tuple[str, ...] = ()
    vision_models: Tuple[str, ...] = ()
    reasoning_effort: str = "low"         # none | low | medium | high (hidden, never shown)
    data_collection: str = "allow"        # OpenRouter: deny = only upstreams that don't store prompts
    llm_temperature: float = 0.1
    max_output_tokens: int = 1500
    llm_timeout: float = 90.0
    image_answers: bool = True            # screen/chart questions: send chart crops to the vision chain
    max_answer_images: int = 2
    image_max_px: int = 1024

    # --- Answering ---------------------------------------------------------------------------
    agent_mode: str = "agent"             # agent (tools only for calculations/comparisons) | quick
    agent_max_steps: int = 3
    observation_chars: int = 1500
    max_context_chars: int = 5000         # ~1,250 evidence tokens per answer (Groq free tier)
    verify_with_llm: bool = False
    max_regenerations: int = 1
    memory_turns: int = 2

    # --- Access and production guards -------------------------------------------------------
    app_password: str = field(default="", repr=False)
    rate_limit_per_min: int = 8
    global_rate_limit_per_min: int = 25
    answer_cache_size: int = 256
    answer_cache_ttl: int = 600           # seconds; keys include the data version, so live data is never stale
    daily_budget_usd: float = 0.0
    enable_feedback: bool = True
    log_level: str = "INFO"

    # --- Live screen capture -----------------------------------------------------------------
    capture_interval_s: float = 2.0
    frame_queue_size: int = 4
    dedup_threshold: float = 0.06         # largest thumbnail-cell change (0..1) below which a frame is a duplicate
    max_capture_pixels: int = 3840 * 2160
    extraction_interval_s: float = 4.0    # OCR + chart reading of the newest frame, at most this often
    ocr_enabled: bool = True
    tesseract_cmd: str = ""
    vision_enabled: bool = True          # background vision-model notes on frames (uses quota)
    vision_interval_s: float = 60.0
    store_screenshots: bool = False       # opt-in replay frames

    # --- Chart model server (Granite Vision 4.1 / ChartAst, local) ---------------------------
    chart_model_enabled: bool = False
    chart_model_url: str = "http://127.0.0.1:8600"
    chart_model_interval_s: float = 30.0
    chart_model_timeout_s: float = 180.0
    chart_model_max_charts: int = 3

    # --- Files and storage -------------------------------------------------------------------
    max_upload_mb: float = 50.0
    max_rows: int = 1_000_000
    max_columns: int = 500
    data_dir: Path = PROJECT_ROOT / "data"
    retention_days: int = 30
    max_observations: int = 2_000_000

    # ----------------------------------------------------------------------------- helpers
    @property
    def db_path(self) -> Path:
        return self.data_dir / "live.db"

    @property
    def screenshot_dir(self) -> Path:
        return self.data_dir / "screenshots"

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key) and not self.local_only

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key) and not self.local_only

    @property
    def has_ollama(self) -> bool:
        return self.ollama_enabled or self.model_profile == "local" or self.local_only

    def provider_configured(self, provider: str) -> bool:
        return {"groq": self.has_groq, "openrouter": self.has_openrouter, "ollama": self.has_ollama}.get(provider, False)

    def models_for(self, role: str) -> Tuple[str, ...]:
        """Ordered "provider:model" chain for a role: explicit override > profile default."""
        from llm.catalog import chain_for

        override = getattr(self, f"{role}_models", ())
        chain = tuple(override) or chain_for("local" if self.local_only else self.model_profile, role)
        if self.ollama_enabled and self.model_profile != "local" and not override:
            chain = chain + chain_for("local", role)  # local fallback when cloud quota runs out
        return chain

    def validate(self) -> "Settings":
        for name, url in (("Groq", self.groq_base_url), ("OpenRouter", self.openrouter_base_url), ("Ollama", self.ollama_base_url), ("Chart model", self.chart_model_url)):
            p = urlparse(url)
            if p.scheme not in ("http", "https") or not p.hostname:
                raise ConfigError(f"{name} URL must look like https://host/path (got {url!r}).")
        if self.local_only and urlparse(self.chart_model_url).hostname not in LOCAL_HOSTS:
            raise ConfigError("Local-only mode is on, so the chart-model server must run on this computer (127.0.0.1 or localhost).")
        if self.model_profile not in ("free", "quality", "local"):
            raise ConfigError("MODEL_PROFILE must be free, quality or local.")
        checks = (
            ("CAPTURE_INTERVAL", self.capture_interval_s, 0.25, 3600), ("EXTRACTION_INTERVAL", self.extraction_interval_s, 0.5, 3600),
            ("VISION_INTERVAL", self.vision_interval_s, 5, 86400), ("FRAME_QUEUE_SIZE", self.frame_queue_size, 1, 64),
            ("DEDUP_THRESHOLD", self.dedup_threshold, 0, 1), ("CHART_MODEL_INTERVAL", self.chart_model_interval_s, 2, 86400),
            ("CHART_MODEL_TIMEOUT", self.chart_model_timeout_s, 5, 3600), ("CHART_MODEL_MAX_CHARTS", self.chart_model_max_charts, 1, 10),
            ("RETENTION_DAYS", self.retention_days, 1, 3650), ("MAX_UPLOAD_MB", self.max_upload_mb, 0.01, 2048),
            ("LLM_TIMEOUT", self.llm_timeout, 5, 600), ("MAX_OUTPUT_TOKENS", self.max_output_tokens, 200, 32000),
        )
        for name, value, lo, hi in checks:
            if not (lo <= value <= hi):
                raise ConfigError(f"{name} must be between {lo} and {hi} (got {value}).")
        return self

    def with_updates(self, **changes: object) -> "Settings":
        return replace(self, **changes).validate()  # type: ignore[arg-type]


def _writable_dir(preferred: Path) -> Path:
    """Use `preferred` if writable, else a temp folder (read-only hosts such as Streamlit Cloud)."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return preferred
    except OSError:
        import tempfile

        fallback = Path(tempfile.gettempdir()) / "poweragent-live-data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Read settings. `env` replaces the environment (used by tests)."""
    s = _Source(env)
    profile = s.choice("MODEL_PROFILE", "free", ("free", "quality", "local"))
    return Settings(
        groq_api_key=(s.lookup("GROQ_API_KEY") or "").strip(),
        groq_base_url=s.str_("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        openrouter_api_key=(s.lookup("OPENROUTER_API_KEY") or "").strip(),
        openrouter_base_url=s.str_("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        ollama_base_url=s.str_("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/"),
        ollama_enabled=s.bool_("OLLAMA_ENABLED", False),
        local_only=s.bool_("LOCAL_ONLY", False),
        app_url=s.str_("APP_URL", "https://github.com/poweragent-live"),
        app_title=s.str_("APP_TITLE", APP_TITLE),
        model_profile=profile,
        master_models=s.models("MASTER_MODELS"),
        fast_models=s.models("FAST_MODELS"),
        verifier_models=s.models("VERIFIER_MODELS"),
        vision_models=s.models("VISION_MODELS"),
        reasoning_effort=s.choice("REASONING_EFFORT", "low", ("none", "low", "medium", "high")),
        data_collection=s.choice("DATA_COLLECTION", "allow", ("allow", "deny")),
        llm_temperature=s.float_("LLM_TEMPERATURE", 0.1, 0.0, 2.0),
        max_output_tokens=s.int_("MAX_OUTPUT_TOKENS", 1500, 200, 32000),
        llm_timeout=s.float_("LLM_TIMEOUT", 90.0, 5.0, 600.0),
        image_answers=s.bool_("IMAGE_ANSWERS", True),
        max_answer_images=s.int_("MAX_ANSWER_IMAGES", 2, 1, 6),
        image_max_px=s.int_("IMAGE_MAX_PX", 1024, 256, 4096),
        agent_mode=s.choice("AGENT_MODE", "agent", ("agent", "quick")),
        agent_max_steps=s.int_("AGENT_MAX_STEPS", 3, 1, 6),
        observation_chars=s.int_("OBSERVATION_CHARS", 1500, 400, 20000),
        max_context_chars=s.int_("MAX_CONTEXT_CHARS", 5000, 1000, 200000),
        verify_with_llm=s.bool_("VERIFY_WITH_LLM", False),
        max_regenerations=s.int_("MAX_REGENERATIONS", 1, 0, 2),
        memory_turns=s.int_("MEMORY_TURNS", 2, 0, 10),
        app_password=(s.lookup("APP_PASSWORD") or "").strip(),
        rate_limit_per_min=s.int_("RATE_LIMIT_PER_MIN", 8, 0, 1000),
        global_rate_limit_per_min=s.int_("GLOBAL_RATE_LIMIT_PER_MIN", 25, 0, 10000),
        answer_cache_size=s.int_("ANSWER_CACHE_SIZE", 256, 0, 10000),
        answer_cache_ttl=s.int_("ANSWER_CACHE_TTL", 600, 0, 86400),
        daily_budget_usd=s.float_("DAILY_BUDGET_USD", 0.0, 0.0, 100000.0),
        enable_feedback=s.bool_("ENABLE_FEEDBACK", True),
        log_level=s.str_("LOG_LEVEL", "INFO").upper(),
        capture_interval_s=s.float_("CAPTURE_INTERVAL", 2.0, 0.25, 3600),
        frame_queue_size=s.int_("FRAME_QUEUE_SIZE", 4, 1, 64),
        dedup_threshold=s.float_("DEDUP_THRESHOLD", 0.06, 0.0, 1.0),
        extraction_interval_s=s.float_("EXTRACTION_INTERVAL", 4.0, 0.5, 3600),
        ocr_enabled=s.bool_("OCR_ENABLED", True),
        tesseract_cmd=s.str_("TESSERACT_CMD", ""),
        vision_enabled=s.bool_("VISION_NOTES", True),
        vision_interval_s=s.float_("VISION_NOTES_INTERVAL", 60.0, 5, 86400),
        store_screenshots=s.bool_("STORE_SCREENSHOTS", False),
        chart_model_enabled=s.bool_("CHART_MODEL_ENABLED", False),
        chart_model_url=s.str_("CHART_MODEL_URL", "http://127.0.0.1:8600").rstrip("/"),
        chart_model_interval_s=s.float_("CHART_MODEL_INTERVAL", 30.0, 2, 86400),
        chart_model_timeout_s=s.float_("CHART_MODEL_TIMEOUT", 180.0, 5, 3600),
        chart_model_max_charts=s.int_("CHART_MODEL_MAX_CHARTS", 3, 1, 10),
        max_upload_mb=s.float_("MAX_UPLOAD_MB", 50.0, 0.01, 2048),
        data_dir=_writable_dir(Path(s.str_("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()),
        retention_days=s.int_("RETENTION_DAYS", 30, 1, 3650),
    ).validate()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER_NAME = "poweragent"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s  %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
