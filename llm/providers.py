"""
Three OpenAI-compatible providers behind one interface (plain httpx, no vendor SDK),
taken over from RAG∞ Ultimate:

    GroqProvider        TEXT: answers, tool calls, verification   (free tier: gpt-oss, Llama)
    OpenRouterProvider  IMAGES: questions about the screen/charts (Gemma 4, Qwen 3.8), key credit info
    OllamaProvider      LOCAL: everything on this PC (MODEL_PROFILE=local), no key

    chat()        POST /chat/completions   (tool calling, images, hidden reasoning control)
    embed()       POST /embeddings         (OpenRouter only)
    rerank()      POST /rerank             (OpenRouter only)
    key_info()    GET  /key                (OpenRouter only: credit limit / usage)
    list_models() GET  /models

Each provider sends only the parameters its API accepts (Groq rejects
OpenRouter's `provider` / `reasoning` objects, so they are never sent there).

Every call returns normalised dataclasses and raises LLMError with a
user-safe message (never containing the key). The model's hidden reasoning
is never stored or returned: the UI shows actions, not chain-of-thought.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from config.settings import Settings, get_logger

log = get_logger("providers")

# Errors after which the router may try the next model in the chain.
FALLBACK_KINDS = {"rate_limit", "connection", "timeout", "model", "api", "quota", "unavailable", "empty"}


class LLMError(Exception):
    """`user_message` is safe to show in the UI (never contains keys)."""

    def __init__(self, user_message: str, kind: str = "error", provider: str = "", status: int = 0):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind
        self.provider = provider
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text


@dataclass
class ChatResult:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tokens: int = 0
    provider: str = "openrouter"
    model: str = ""            # model that actually answered (as reported by OpenRouter)
    cost: float = 0.0          # USD, from usage.cost
    upstream: str = ""         # upstream provider that served it (e.g. "Alibaba", "Google AI Studio")
    finish_reason: str = ""


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    tokens: int = 0
    cost: float = 0.0


@dataclass
class KeyInfo:
    label: str = ""
    usage: float = 0.0             # USD spent with this key
    limit: Optional[float] = None  # USD limit on the key (None = unlimited)
    remaining: Optional[float] = None
    is_free_tier: bool = False


_CLIENTS: Dict[tuple, httpx.Client] = {}
_CLIENTS_LOCK = threading.Lock()


def _client(base_url: str, timeout: float) -> httpx.Client:
    key = (base_url, timeout)
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(key)
        if c is None:
            c = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=15.0),
                             limits=httpx.Limits(max_connections=32, max_keepalive_connections=16))
            _CLIENTS[key] = c
        return c


def _png_data_url(data: bytes) -> str:
    mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class OpenRouterProvider:
    name, label = "openrouter", "OpenRouter"

    key_env, key_url = "OPENROUTER_API_KEY", "openrouter.ai/settings/keys"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url

    def configured(self) -> bool:
        return bool(self.api_key) and not self.settings.local_only

    # ------------------------------------------------------------ transport
    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError(f"No {self.label} API key is set. Add {self.key_env} to .env or the app's secrets "
                           f"(create one at {self.key_url}).", "auth", self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "HTTP-Referer": self.settings.app_url, "X-Title": self.settings.app_title}

    def _request(self, method: str, path: str, body: Optional[Dict] = None, model: str = "",
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        """One HTTP call with a single retry for transient upstream failures (502/503/529)."""
        client = _client(self.base_url, timeout or self.settings.llm_timeout)
        last: Optional[LLMError] = None
        for attempt in range(2):
            try:
                resp = client.request(method, path, json=body, headers=self._headers())
            except httpx.TimeoutException as exc:
                raise LLMError(f"{self.label} took too long to respond.", "timeout", self.name) from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Cannot reach {self.label} right now. Check the internet connection.",
                               "connection", self.name) from exc
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise LLMError(f"{self.label} returned an unreadable response.", "api", self.name) from exc
                if isinstance(data, dict) and data.get("error") and not data.get("choices") \
                        and not data.get("data") and not data.get("results"):
                    raise self._translate(int(data["error"].get("code") or 502), data, model)
                return data
            try:
                payload = resp.json()
            except ValueError:
                payload = {"error": {"message": resp.text[:300]}}
            last = self._translate(resp.status_code, payload, model)
            last.provider = self.name
            if resp.status_code in (502, 503, 529) and attempt == 0:
                time.sleep(1.2)
                continue
            raise last
        assert last is not None
        raise last

    @staticmethod
    def _translate(status: int, payload: Dict, model: str) -> LLMError:
        err = (payload or {}).get("error") or {}
        msg = str(err.get("message") or "")[:300]
        meta = err.get("metadata") or {}
        name = model or "the model"
        if status == 401:
            return LLMError("OpenRouter rejected the API key. Create a new one at openrouter.ai/settings/keys.",
                            "auth", status=status)
        if status == 402:
            return LLMError(f"Not enough OpenRouter credits for '{name}'. Add credits or switch to the Free "
                            "profile.", "quota", status=status)
        if status == 403:
            reason = ", ".join(meta.get("reasons", [])) if isinstance(meta.get("reasons"), list) else ""
            return LLMError(f"'{name}' refused this request (moderation{': ' + reason if reason else ''}).",
                            "moderation", status=status)
        if status == 404:
            return LLMError(f"'{name}' is not available on OpenRouter (or no provider supports the requested "
                            "features).", "model", status=status)
        if status == 408:
            return LLMError("OpenRouter timed out.", "timeout", status=status)
        if status == 413:
            return LLMError("The request was too large for the model.", "bad_request", status=status)
        if status == 429:
            free = ":free" in name
            return LLMError(f"Rate limit reached for '{name}'" + (" (free endpoints allow ~20 requests/min)."
                            if free else "."), "rate_limit", status=status)
        if status == 400:
            low = msg.lower()
            if "tool" in low and ("call" in low or "parse" in low or "invalid" in low):
                return LLMError("The model produced an invalid tool call.", "tool_use_failed", status=status)
            if "context" in low and ("length" in low or "too long" in low):
                return LLMError("The question plus evidence is too long for the model.", "bad_request",
                                status=status)
            log.warning("OpenRouter rejected a request for %s: %s", name, msg)
            return LLMError("OpenRouter rejected the request.", "bad_request", status=status)
        if status in (502, 503, 529):
            return LLMError(f"'{name}' is temporarily unavailable upstream.", "unavailable", status=status)
        log.warning("OpenRouter error %s for %s: %s", status, name, msg)
        return LLMError("OpenRouter returned an error.", "api", status=status)

    # ----------------------------------------------------------------- chat
    def _body(self, model: str, messages: List[Dict[str, Any]], tools, tool_choice: str, temperature: float,
              max_tokens: int, reasoning: Optional[str], json_mode: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        effort = reasoning or self.settings.reasoning_effort
        # Hidden reasoning: bounded effort, and never returned to us (exclude).
        body["reasoning"] = {"enabled": False} if effort == "none" else {"effort": effort, "exclude": True}
        provider: Dict[str, Any] = {"allow_fallbacks": True}
        if self.settings.data_collection == "deny":
            provider["data_collection"] = "deny"
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
            provider["require_parameters"] = True  # only route to upstreams that really support tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body["provider"] = provider
        return body

    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", temperature: float = 0.1, max_tokens: int = 1500,
             images: Optional[Sequence[bytes]] = None, reasoning: Optional[str] = None,
             json_mode: bool = False) -> ChatResult:
        if images:  # attach images to the last user message (OpenAI vision format)
            messages = [dict(m) for m in messages]
            last = messages[-1]
            parts: List[Dict[str, Any]] = [{"type": "text", "text": last.get("content", "")}]
            parts += [{"type": "image_url", "image_url": {"url": _png_data_url(b)}} for b in images]
            last["content"] = parts
        body = self._body(model, messages, tools, tool_choice, temperature, max_tokens, reasoning, json_mode)
        data = self._request("POST", "/chat/completions", body, model)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"'{model}' returned no answer.", "empty")
        choice = choices[0]
        if choice.get("error"):
            raise self._translate(int(choice["error"].get("code") or 502), {"error": choice["error"]}, model)
        msg = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or "{}"
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        content = msg.get("content") or ""
        if isinstance(content, list):  # some upstreams return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = data.get("usage") or {}
        if not content.strip() and not calls:
            raise LLMError(f"'{model}' returned an empty answer.", "empty")
        return ChatResult(content=content.strip(), tool_calls=calls, tokens=int(usage.get("total_tokens") or 0),
                          provider=self.name, model=str(data.get("model") or model),
                          cost=float(usage.get("cost") or 0.0),
                          upstream=str(data.get("provider") or ""), finish_reason=str(choice.get("finish_reason") or ""))

    # ------------------------------------------------------- embeddings
    def embed(self, model: str, texts: Sequence[str], dimensions: Optional[int] = None) -> EmbedResult:
        body: Dict[str, Any] = {"model": model, "input": list(texts), "encoding_format": "float"}
        if dimensions:
            body["dimensions"] = int(dimensions)
        if self.settings.data_collection == "deny":
            body["provider"] = {"data_collection": "deny"}
        data = self._request("POST", "/embeddings", body, model, timeout=max(60.0, self.settings.llm_timeout))
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        vectors = []
        for r in rows:
            emb = r.get("embedding")
            if isinstance(emb, str):  # base64 float32
                import numpy as np

                emb = np.frombuffer(base64.b64decode(emb), dtype="<f4").tolist()
            vectors.append([float(x) for x in emb or []])
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise LLMError("The embedding model returned an incomplete result.", "api")
        usage = data.get("usage") or {}
        return EmbedResult(vectors, int(usage.get("total_tokens") or 0), float(usage.get("cost") or 0.0))

    # ----------------------------------------------------------- rerank
    def rerank(self, model: str, query: str, documents: Sequence[str]) -> List[float]:
        """Relevance score (0..1) for every document, in input order."""
        body: Dict[str, Any] = {"model": model, "query": query, "documents": list(documents),
                                "top_n": len(documents)}
        data = self._request("POST", "/rerank", body, model, timeout=30.0)
        scores = [0.0] * len(documents)
        for r in data.get("results") or []:
            idx = int(r.get("index", -1))
            if 0 <= idx < len(scores):
                scores[idx] = float(r.get("relevance_score") or 0.0)
        return scores

    # ------------------------------------------------------- account info
    def key_info(self) -> KeyInfo:
        d = (self._request("GET", "/key", None, timeout=15.0) or {}).get("data") or {}
        limit = d.get("limit")
        remaining = d.get("limit_remaining")
        return KeyInfo(label=str(d.get("label") or ""), usage=float(d.get("usage") or 0.0),
                       limit=None if limit is None else float(limit),
                       remaining=None if remaining is None else float(remaining),
                       is_free_tier=bool(d.get("is_free_tier")))

    def raw_models(self) -> List[Dict[str, Any]]:
        """OpenRouter's public model catalogue (several MB). Fetched with plain gzip and a long timeout,
        retried once: some networks / antivirus proxies cut large compressed responses."""
        last: Optional[Exception] = None
        for timeout in (60.0, 90.0):
            try:
                r = httpx.get(self.base_url.rstrip("/") + "/models", timeout=httpx.Timeout(timeout, connect=15.0),
                              headers={"Accept-Encoding": "gzip", "Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                return list(r.json().get("data") or [])
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
        raise LLMError(f"Could not download the OpenRouter model list ({type(last).__name__}). Model calls can "
                       "still work - this list is only used for checks.", "connection", "openrouter")

    def list_models(self, families: Sequence[str] = ("qwen/", "google/gemma")) -> List[str]:
        ids = [str(m.get("id", "")) for m in self.raw_models()]
        return sorted(i for i in ids if i and (not families or any(i.startswith(f) for f in families)))


# =============================================================================
# Groq (text): free tier, very fast, OpenAI-compatible
# =============================================================================
_GROQ_NON_CHAT = ("whisper", "orpheus", "guard", "tts", "playai", "distil", "safeguard")


class GroqProvider(OpenRouterProvider):
    """Same transport and response parsing; Groq-specific request body and errors.

    Groq's free tier reports no cost (cost = 0). Hidden reasoning of gpt-oss models
    is kept short (`reasoning_effort`) and never returned (`include_reasoning: false`)."""

    name, label = "groq", "Groq"
    key_env, key_url = "GROQ_API_KEY", "console.groq.com/keys"

    def configured(self) -> bool:
        return bool(self.api_key) and not self.settings.local_only

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.groq_api_key
        self.base_url = settings.groq_base_url

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError("No Groq API key is set. Add GROQ_API_KEY to .env or the app's secrets "
                           "(create a free key at console.groq.com/keys).", "auth", self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, model, messages, tools, tool_choice, temperature, max_tokens, reasoning, json_mode):
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_completion_tokens": max_tokens}
        if model.startswith("openai/gpt-oss"):
            effort = reasoning or self.settings.reasoning_effort
            body["reasoning_effort"] = "low" if effort == "none" else effort
            body["include_reasoning"] = False
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _translate(status: int, payload: Dict, model: str) -> LLMError:
        err = (payload or {}).get("error") or {}
        code = str(err.get("code") or "")
        msg = str(err.get("message") or "")[:300]
        name = model or "the model"
        if status == 401:
            return LLMError("Groq rejected the API key. Create a new one at console.groq.com/keys.", "auth",
                            "groq", status)
        if status == 413 or "rate_limit" in code or status == 429:
            return LLMError(f"Groq's free limit was reached for '{name}' (requests or tokens per minute/day).",
                            "rate_limit", "groq", status)
        if status == 404 or code == "model_not_found" or "decommissioned" in msg.lower():
            return LLMError(f"'{name}' is not available on Groq.", "model", "groq", status)
        if status == 400 and (code == "tool_use_failed" or "tool call" in msg.lower()):
            return LLMError("The model produced an invalid tool call.", "tool_use_failed", "groq", status)
        if status == 400 and "context_length" in (code + msg.lower()):
            return LLMError("The question plus evidence is too long for the model.", "bad_request", "groq", status)
        if status == 400:
            log.warning("Groq rejected a request for %s: %s", name, msg)
            return LLMError("Groq rejected the request.", "bad_request", "groq", status)
        if status in (498, 500, 502, 503):
            return LLMError(f"Groq is temporarily unavailable for '{name}'.", "unavailable", "groq", status)
        return LLMError("Groq returned an error.", "api", "groq", status)

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        data = self._request("GET", "/models", None, timeout=20.0)
        return sorted(str(m.get("id")) for m in data.get("data") or []
                      if m.get("id") and not any(h in str(m["id"]).lower() for h in _GROQ_NON_CHAT))

    def key_info(self) -> KeyInfo:  # Groq has no key-info endpoint; listing models proves the key works
        self.list_models()
        return KeyInfo(label="groq", is_free_tier=True)

    def embed(self, *a, **kw):  # pragma: no cover - not offered by this app on Groq
        raise LLMError("Embeddings are not used on Groq.", "bad_request", "groq")

    def rerank(self, *a, **kw):  # pragma: no cover
        raise LLMError("Reranking is not used on Groq.", "bad_request", "groq")


# =============================================================================
# Ollama (local): OpenAI-compatible endpoint of the Ollama app, no key, nothing leaves the PC
# =============================================================================
class OllamaProvider(OpenRouterProvider):
    """Ollama's /v1 API (chat with tools and images). Short connect timeout so a stopped
    Ollama makes the router move on quickly instead of hanging."""

    name, label = "ollama", "Ollama"
    key_env, key_url = "OLLAMA_ENABLED", "ollama.com/download"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = ""
        self.base_url = settings.ollama_base_url

    def configured(self) -> bool:
        return self.settings.has_ollama

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _request(self, method: str, path: str, body: Optional[Dict] = None, model: str = "",
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        client = _ollama_client(self.base_url, timeout or self.settings.llm_timeout)
        try:
            resp = client.request(method, path, json=body, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama took too long to respond.", "timeout", self.name) from exc
        except httpx.HTTPError as exc:
            raise LLMError("Cannot reach Ollama on this computer. Start the Ollama app (or `ollama serve`).",
                           "connection", self.name) from exc
        if resp.status_code < 400:
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError("Ollama returned an unreadable response.", "api", self.name) from exc
        try:
            payload = resp.json()
        except ValueError:
            payload = {"error": {"message": resp.text[:300]}}
        raise self._translate(resp.status_code, payload, model)

    @staticmethod
    def _translate(status: int, payload: Dict, model: str) -> LLMError:
        err = (payload or {}).get("error") or {}
        msg = str(err.get("message") if isinstance(err, dict) else err)[:300]
        name = model or "the model"
        if status == 404 or "not found" in msg.lower():
            return LLMError(f"'{name}' is not installed in Ollama. Run: ollama pull {name}", "model", "ollama", status)
        if "does not support tools" in msg.lower():
            return LLMError(f"'{name}' does not support tool calling.", "tool_use_failed", "ollama", status)
        if status == 400:
            return LLMError(f"Ollama rejected the request: {msg}", "bad_request", "ollama", status)
        return LLMError(f"Ollama error {status}: {msg}", "unavailable", "ollama", status)

    def _body(self, model, messages, tools, tool_choice, temperature, max_tokens, reasoning, json_mode):
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        data = self._request("GET", "/models", None, timeout=10.0)
        return sorted(str(m.get("id")) for m in data.get("data") or [] if m.get("id"))

    def key_info(self) -> KeyInfo:
        self.list_models()
        return KeyInfo(label="local", is_free_tier=True)

    def embed(self, *a, **kw):  # pragma: no cover
        raise LLMError("Embeddings are not used.", "bad_request", "ollama")

    def rerank(self, *a, **kw):  # pragma: no cover
        raise LLMError("Reranking is not used.", "bad_request", "ollama")


_OLLAMA_CLIENTS: Dict[tuple, httpx.Client] = {}


def _ollama_client(base_url: str, timeout: float) -> httpx.Client:
    key = (base_url, timeout)
    with _CLIENTS_LOCK:
        c = _OLLAMA_CLIENTS.get(key)
        if c is None:
            c = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=2.0))
            _OLLAMA_CLIENTS[key] = c
        return c


def make_provider(name: str, settings: Settings):
    if name == "groq":
        return GroqProvider(settings)
    if name == "ollama":
        return OllamaProvider(settings)
    return OpenRouterProvider(settings)
