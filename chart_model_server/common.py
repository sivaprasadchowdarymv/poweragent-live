"""Shared HTTP layer for chart-model inference servers (Granite Vision, ChartAst).

API (identical for every engine, so the app does not care which model runs):

    GET  /health
    POST /v1/chart_to_table   {"image": "<base64 PNG/JPEG>"}             -> {"text": table, "format": "csv"|"markdown"}
    POST /v1/chart_qa         {"image": "...", "question": "..."}         -> {"text": answer}
    POST /v1/summarize        {"image": "..."}                            -> {"text": summary}

Standard library only, so it can be tested without torch. Engines import
their ML stack lazily.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

log = logging.getLogger("chart_model_server")

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_QUESTION_CHARS = 500
ROUTES = {"/v1/chart_to_table": "table", "/v1/chart_qa": "qa", "/v1/summarize": "summary"}


class Engine(Protocol):
    name: str
    info: dict[str, Any]
    output_format: str  # format of the 'table' task output: "csv" or "markdown"

    def run(self, task: str, image: Any, question: str, max_new_tokens: int, temperature: float) -> str: ...


class ChartModelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], engine: Engine, temperature: float = 0.0, busy_timeout_s: float = 300.0) -> None:
        super().__init__(addr, _Handler)
        self.engine = engine
        self.temperature = temperature
        self.busy_timeout_s = busy_timeout_s
        self.gpu_lock = threading.Lock()  # one generation at a time
        self.stats: dict[str, Any] = {"requests": 0, "errors": 0, "last_ms": None}


class _Handler(BaseHTTPRequestHandler):
    server: ChartModelHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            s = self.server
            return self._send(200, {"ok": True, "busy": s.gpu_lock.locked(), "output_format": s.engine.output_format, **s.engine.info, **s.stats})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        task = ROUTES.get(self.path.rstrip("/"))
        if task is None:
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_IMAGE_BYTES * 2:
            return self._send(400, {"error": "Request body missing or too large."})
        try:
            body = json.loads(self.rfile.read(length))
            image = decode_image(body.get("image", ""))
        except (ValueError, KeyError, OSError) as exc:
            return self._send(400, {"error": f"Invalid request: {exc}"})
        question = str(body.get("question", ""))[:MAX_QUESTION_CHARS]
        if task == "qa" and not question.strip():
            return self._send(400, {"error": "chart_qa needs a 'question'."})
        try:
            max_new = int(body.get("max_new_tokens", 2048 if task == "table" else 512))
            temperature = float(body.get("temperature", self.server.temperature))
        except (TypeError, ValueError):
            return self._send(400, {"error": "max_new_tokens/temperature must be numbers."})
        max_new = max(16, min(max_new, 4096))
        s = self.server
        if not s.gpu_lock.acquire(timeout=s.busy_timeout_s):
            return self._send(503, {"error": "The model is busy with another request; try again later."})
        t0 = time.perf_counter()
        try:
            s.stats["requests"] += 1
            text = s.engine.run(task, image, question, max_new, temperature)
        except Exception as exc:  # report, never crash the server
            s.stats["errors"] += 1
            log.exception("generation failed")
            return self._send(500, {"error": f"Generation failed: {exc}"})
        finally:
            s.gpu_lock.release()
        ms = (time.perf_counter() - t0) * 1000
        s.stats["last_ms"] = round(ms)
        self._send(200, {"task": task, "text": text, "format": s.engine.output_format if task == "table" else "text", "ms": round(ms), "model": s.engine.name})


def decode_image(b64: str) -> Any:
    from PIL import Image

    if not isinstance(b64, str) or not b64:
        raise ValueError("'image' must be a base64-encoded PNG or JPEG.")
    if b64.startswith("data:") and "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64, validate=False)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than 12 MB.")
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.width * img.height > 40_000_000:
        raise ValueError("Image has too many pixels.")
    return img.convert("RGB")


def serve(engine: Engine, host: str, port: int, temperature: float = 0.0) -> None:
    srv = ChartModelHTTPServer((host, port), engine, temperature)
    log.info("Serving %s on http://%s:%d  (Ctrl+C to stop)", engine.name, host, port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        srv.server_close()
