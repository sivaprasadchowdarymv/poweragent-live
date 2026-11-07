"""
Check your Groq / OpenRouter / Ollama, OCR and capture setup before running the app.

    python -m scripts.check_setup            # key, credits, every model id, one tiny call per role
    python -m scripts.check_setup --no-calls # only key + catalogue checks (costs nothing)

It verifies that:
  1. GROQ_API_KEY (text) and OPENROUTER_API_KEY (images) work;
  2. every model in your chains exists in that provider's live catalogue
     (model ids change when new versions ship);
  3. each role really answers: master with a tool call (Groq), fast and verifier (Groq),
     vision with an image (Gemma / Qwen).
With the free profile the full check costs $0.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from llm.catalog import ROLES, split  # noqa: E402
from llm.model_router import ModelRouter  # noqa: E402
from llm.providers import GroqProvider, LLMError, OllamaProvider, OpenRouterProvider  # noqa: E402
from agents.tools import default_registry  # noqa: E402

TOOL_SCHEMAS = [default_registry().tools["calculate_statistics"].schema()]

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def _png() -> bytes:
    """A 64x32 PNG with a black bar (enough for a vision smoke test)."""
    w, h = 64, 32
    rows = b"".join(b"\x00" + b"".join(b"\x00\x00\x00" if 8 < y < 24 and 8 < x < 56 else b"\xff\xff\xff"
                                      for x in range(w)) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _local_checks(s) -> None:
    from vision.ocr import configure_tesseract, ocr_status

    configure_tesseract(s.tesseract_cmd or None)
    ok, msg = ocr_status()
    print(f"{OK if ok else WARN} Tesseract OCR: {msg}")
    try:
        from capture.screen import list_monitors

        print(f"{OK} screen capture: {len(list_monitors())} monitor(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} screen capture unavailable here ({exc}); Data + Chat still work")
    if s.chart_model_enabled:
        from vision.chart_model import ChartModelClient

        h = ChartModelClient(s.chart_model_url, timeout=5).health()
        print(f"{OK if h.get('ok') else WARN} chart model server {s.chart_model_url}: {h.get('model') or h.get('error')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-calls", action="store_true", help="skip the paid test calls")
    a = ap.parse_args()
    s = load_settings()
    failures = 0
    print(f"Profile: {s.model_profile}  (text -> Groq, screen images -> Gemma/Qwen on OpenRouter, local -> Ollama)")
    _local_checks(s)
    if not s.groq_api_key and s.model_profile != "local" and not s.local_only:
        failures += 1
        print(f"{BAD} GROQ_API_KEY is not set: text answers need it (free key at console.groq.com/keys).")
    if not s.openrouter_api_key:
        print(f"{WARN} OPENROUTER_API_KEY is not set: screen questions use OCR/chart-reader evidence only "
              "(no image reading (free key at openrouter.ai/settings/keys).")
    catalogues = {}
    for name, provider in (("groq", GroqProvider(s)), ("openrouter", OpenRouterProvider(s)), ("ollama", OllamaProvider(s))):
        if not provider.configured():
            continue
        try:
            k = provider.key_info()
            extra = ""
            if name == "openrouter":
                limit = "no limit (set one at openrouter.ai/settings/keys)" if k.limit is None else f"${k.limit:.2f}"
                extra = f" · used ${k.usage:.4f} · limit {limit}"
            print(f"{OK} {provider.label} key works{extra}")
        except LLMError as exc:
            failures += 1
            print(f"{BAD} {provider.label} key check failed: {exc.user_message}")
            continue
        try:
            catalogues[name] = set(provider.list_models(families=()))
        except LLMError as exc:
            print(f"{WARN} {provider.label} model list unavailable ({exc.user_message}); the live test calls below decide.")
    wanted = {r: s.models_for(r) for r in ROLES}
    keyed = {"groq": GroqProvider(s).configured(), "openrouter": OpenRouterProvider(s).configured(),
             "ollama": OllamaProvider(s).configured()}
    for role, entries in wanted.items():
        usable = 0
        for entry in entries:
            name, model = split(entry)
            cat = catalogues.get(name)
            if not keyed.get(name):
                print(f"{WARN} {role:<9} {entry} (skipped: no {name} key)")
            elif cat is None:
                usable += 1
                print(f"{WARN} {role:<9} {entry} (not checked: model list unavailable)")
            elif model in cat:
                usable += 1
                print(f"{OK} {role:<9} {entry}")
            else:
                print(f"{WARN} {role:<9} {entry} is not offered any more - the app skips it automatically")
        if usable == 0 and any(keyed.get(split(e)[0]) for e in entries):
            failures += 1
            print(f"{BAD} {role:<9} no usable model in this chain - send this output to get the model list updated")
    if "groq" in catalogues:
        print("   Groq models on your account: " + ", ".join(sorted(m for m in catalogues["groq"] if "whisper" not in m)[:20]))
    if catalogues.get("openrouter"):
        free_vis = [m for m in catalogues["openrouter"] if m.endswith(":free") and ("gemma" in m or "qwen" in m)]
        print("   Free Gemma/Qwen models on OpenRouter: " + (", ".join(sorted(free_vis)) or "none listed"))
    if a.no_calls or failures:
        return 1 if failures else 0
    r = ModelRouter(s)
    tests = {
        "master": dict(messages=[{"role": "user", "content": "Call calculate_statistics for metric current_a."}],
                       tools=TOOL_SCHEMAS, max_tokens=300),
        "fast": dict(messages=[{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20),
        "verifier": dict(messages=[{"role": "user", "content": 'Reply with JSON only: {"grounded": true}'}],
                         max_tokens=50, json_mode=True),
        "vision": dict(messages=[{"role": "user", "content": "What shape and colour is in this image? One sentence."}],
                       images=[_png()], max_tokens=60),
    }
    total = 0.0
    for role, kw in tests.items():
        if not r.has_role(role):
            print(f"{WARN} {role:<9} skipped (no key for its provider)")
            continue
        try:
            out = r.chat(role, **kw)
            res = out.result
            total += res.cost
            what = (f"tool call {res.tool_calls[0].name}({res.tool_calls[0].arguments})" if res.tool_calls
                    else json.dumps(res.content[:60]))
            fb = "" if out.attempts[0][2] == "ok" else f"  (fallback after {out.attempts[0][1]}: {out.attempts[0][2]})"
            print(f"{OK} {role:<9} {res.provider}:{res.model} -> {what} · ${res.cost:.5f}{fb}")
            if role == "master" and not res.tool_calls:
                print(f"{WARN} master answered without calling the tool; agent mode still works but uses fewer tools")
        except LLMError as exc:
            failures += 1
            print(f"{BAD} {role:<9} {exc.user_message}")
    print(f"\nTotal cost of this check: ${total:.5f}")
    print("All good." if not failures else f"{failures} problem(s) found.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
