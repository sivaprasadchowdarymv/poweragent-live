"""Process-wide runtime: one database, one capture worker, one extraction worker.

Streamlit reruns the script on every interaction; holding these objects in a
single cached Runtime (``st.cache_resource`` in app.py) guarantees reruns
never spawn duplicate threads. The app is intended for a single local user.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from capture.screen import MSSScreenSource, ScreenSource
from capture.worker import CaptureWorker, Frame
from agents.tools import ToolContext, default_registry
from analysis.alerts import AlertEngine
from config import Settings
from llm.model_router import ModelRouter
from core.models import utcnow
from core.services import DataService
from storage.db import Database
from vision.chart_model import ChartModelClient, read_frame
from vision.chart_reader import ScreenAnalysis
from vision.extraction import ExtractionWorker
from vision.vision_model import describe_screen

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings: Settings, source_factory: Any = MSSScreenSource, router: ModelRouter | None = None, db: Database | None = None) -> None:
        self.settings = settings
        self.db = db or Database(settings.db_path)
        self.service = DataService(self.db)
        self.alerts = AlertEngine(self.db)
        self.alert_events: list[dict[str, Any]] = []
        self.router = router or ModelRouter(settings)
        self._custom_router = router is not None
        self.registry = default_registry()
        self.chat_lock = threading.Lock()  # one answer at a time (protects the free-tier quota)
        self.capture = CaptureWorker(
            source_factory,
            interval_s=settings.capture_interval_s,
            queue_size=settings.frame_queue_size,
            dedup_threshold=settings.dedup_threshold,
            max_pixels=settings.max_capture_pixels,
        )
        self.extraction = ExtractionWorker(
            self.capture,
            on_result=self._on_analysis,
            interval_s=settings.extraction_interval_s,
            vision_call=self._vision_call,
            on_vision=self._on_vision,
            vision_interval_s=settings.vision_interval_s,
            on_frame=self._maybe_store_frame,
            chart_model_call=self._chart_model_call,
            on_chart_model=self._on_chart_model,
            chart_model_interval_s=settings.chart_model_interval_s,
        )
        self.chart_model = ChartModelClient(settings.chart_model_url, timeout=settings.chart_model_timeout_s)
        self.extraction.chart_model_enabled = settings.chart_model_enabled
        self.alerts.load(self.rules_path)
        self.benchmark_proc: Any = None
        self.reload_priors()
        self.extraction.vision_enabled = settings.vision_enabled
        self.extraction.ocr_enabled = settings.ocr_enabled
        self._last_retention = 0.0
        self.analysis_status = "idle"
        self.browser_frame: Frame | None = None
        self._browser_seq = 0
        self.vision_backoff_until = 0.0
        self.vision_backoff_s = 60.0
        self.vision_last_error = ""

    @property
    def rules_path(self) -> Any:
        return self.settings.data_dir / "alert_rules.json"

    def save_rules(self) -> None:
        self.alerts.save(self.rules_path)

    def _maybe_store_frame(self, frame: Frame, analysis: ScreenAnalysis) -> None:
        """Opt-in only: keep a bounded set of JPEG frames for replay."""
        if not self.settings.store_screenshots:
            return
        d = self.settings.screenshot_dir
        d.mkdir(parents=True, exist_ok=True)
        name = f"{frame.captured_at.strftime('%Y%m%dT%H%M%S_%f')}_{frame.seq}.jpg"
        img = frame.image.convert("RGB")
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600))
        img.save(d / name, quality=80)
        files = sorted(d.glob("*.jpg"))
        for old in files[:-300]:
            old.unlink(missing_ok=True)

    def stored_frames(self) -> list[Any]:
        d = self.settings.screenshot_dir
        return sorted(d.glob("*.jpg")) if d.exists() else []

    def delete_stored_frames(self) -> int:
        n = 0
        for f in self.stored_frames():
            f.unlink(missing_ok=True)
            n += 1
        return n

    def reload_priors(self) -> dict[str, Any]:
        from metrics.priors import load_priors

        self.service.priors = load_priors(self.settings.data_dir)
        return self.service.priors

    def start_benchmark(self, samples: int, readers: str) -> str:
        """Run the ChartNet benchmark in a separate process (needs the free 'datasets' package and internet)."""
        import subprocess
        import sys

        if self.benchmark_proc is not None and self.benchmark_proc.poll() is None:
            return "A benchmark is already running."
        log_dir = self.settings.data_dir / "benchmarks"
        log_dir.mkdir(parents=True, exist_ok=True)
        root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
        cmd = [sys.executable, "-m", "metrics.chartnet_eval", "--samples", str(int(samples)), "--readers", readers, "--model-url", self.settings.chart_model_url]
        env = dict(__import__("os").environ, DATA_DIR=str(self.settings.data_dir), PYTHONIOENCODING="utf-8")
        self._bench_log = log_dir / "last_run.log"
        fh = open(self._bench_log, "w", encoding="utf-8")  # noqa: SIM115 - closed when the process ends
        self.benchmark_proc = subprocess.Popen(cmd, cwd=root, stdout=fh, stderr=subprocess.STDOUT, env=env)
        return "Benchmark started."

    def benchmark_status(self) -> dict[str, Any]:
        p = self.benchmark_proc
        log = ""
        if getattr(self, "_bench_log", None) and self._bench_log.exists():
            log = self._bench_log.read_text(encoding="utf-8", errors="replace")[-3000:]
        if p is None:
            return {"state": "idle", "log": log}
        code = p.poll()
        if code is None:
            return {"state": "running", "log": log}
        return {"state": "finished" if code == 0 else f"failed (exit {code})", "log": log}

    def apply_settings(self, settings: Settings) -> None:
        old = self.settings
        self.settings = settings
        if not self._custom_router:
            self.router = ModelRouter(settings)
        self.capture.configure(interval_s=settings.capture_interval_s)
        self.capture.dedup_threshold = settings.dedup_threshold
        self.extraction.interval_s = settings.extraction_interval_s
        self.extraction.vision_interval_s = settings.vision_interval_s
        self.extraction.vision_enabled = settings.vision_enabled
        self.extraction.ocr_enabled = settings.ocr_enabled
        if (settings.chart_model_url, settings.chart_model_timeout_s) != (old.chart_model_url, old.chart_model_timeout_s):
            self.chart_model = ChartModelClient(settings.chart_model_url, timeout=settings.chart_model_timeout_s)
        self.extraction.chart_model_enabled = settings.chart_model_enabled
        self.extraction.chart_model_interval_s = settings.chart_model_interval_s

    # ------------------------------------------------------------ monitoring
    def start_monitoring(self) -> None:
        self.capture.start()
        self.extraction.start()

    def stop_monitoring(self) -> None:
        self.extraction.stop()
        self.capture.stop()

    def manual_capture(self) -> ScreenAnalysis:
        frame: Frame | None = None
        if self.capture.state == "running":
            self.capture.request_capture()
            frame = self.capture.wait_for_frame(self.extraction.metrics.last_frame_seq, timeout=3.0) or self.capture.latest()
        if frame is None:
            frame = self.capture.capture_once()
        return self.extraction.process_now(frame)

    def _on_analysis(self, analysis: ScreenAnalysis) -> None:
        self.analysis_status = "analyzing"
        try:
            self.service.add_screen_analysis(analysis)
            if self.alerts.rules:
                events = self.alerts.evaluate()
                self.alert_events = (self.alert_events + [e | {"at": utcnow().isoformat()} for e in events if e["event"] in ("opened", "resolved")])[-50:]
            import time as _t

            if _t.monotonic() - self._last_retention > 600:
                self.db.apply_retention(self.settings.retention_days, self.settings.max_observations)
                self._last_retention = _t.monotonic()
        finally:
            self.analysis_status = "idle"

    def _chart_model_call(self, frame: Frame, analysis: ScreenAnalysis) -> list[Any]:
        return read_frame(self.chart_model, frame.image, analysis.charts, self.settings.chart_model_max_charts)

    def _on_chart_model(self, readings: list[Any], frame: Frame, analysis: ScreenAnalysis) -> None:
        self.service.add_chart_model_readings(readings, frame.seq, frame.captured_at)

    def chart_model_now(self) -> bool:
        """Manual chart-model reading of the latest frame (background; returns False if one is running)."""
        frame = self.capture.latest()
        analysis = self.service.latest_screen
        if frame is None or analysis is None or analysis.frame_seq != frame.seq:
            analysis = self.manual_capture()
            frame = self.capture.latest()
        if frame is None:
            return False
        return self.extraction.request_chart_model(frame, analysis) or self.extraction.chart_model_busy

    def _vision_call(self, frame: Frame) -> dict[str, Any]:
        """Background vision reading of a changed frame. After rate limits it backs off (1 -> 15 min),
        so the free daily quota is not burned while the models are busy."""
        import time as _t

        from llm.providers import LLMError

        if _t.monotonic() < self.vision_backoff_until:
            raise RuntimeError("vision backoff after rate limit")
        try:
            out = describe_screen(self.router, frame.image, max(self.settings.image_max_px, 1280))
            self.vision_backoff_s = 60.0
            self.vision_last_error = ""
            return out
        except LLMError as exc:
            self.vision_last_error = exc.user_message
            if exc.kind in ("rate_limit", "quota", "unavailable", "timeout"):
                self.vision_backoff_until = _t.monotonic() + self.vision_backoff_s
                self.vision_backoff_s = min(self.vision_backoff_s * 2, 900.0)
            raise

    def analyze_image_now(self, image: Any, label: str = "uploaded image") -> dict[str, Any]:
        """Uploaded image or on-demand reading: OCR + chart reader + vision model, synchronously."""
        import time as _t

        from capture.screen import Region
        from capture.worker import fingerprint

        self._browser_seq += 1
        frame = Frame(1_000_000 + self._browser_seq, utcnow(), _t.monotonic(), image.convert("RGB"), Region(0, 0, *image.size), label, fingerprint(image))
        self.browser_frame = frame
        self.extraction.process_now(frame)
        notes = describe_screen(self.router, frame.image, max(self.settings.image_max_px, 1280))
        self.service.set_vision_notes(notes, frame.seq, frame.captured_at)
        return notes

    def _on_vision(self, notes: dict[str, Any], frame: Frame) -> None:
        self.service.set_vision_notes(notes, frame.seq, frame.captured_at)

    # ------------------------------------------------------------ chat
    def tool_context(self, focus: dict[str, Any] | None = None, conversation_id: str | None = None) -> ToolContext:
        """Context for the controlled analysis tools of one chat turn."""
        ctx = ToolContext(self.service, self.alerts, conversation_id, focus if focus is not None else {})
        if self.settings.chart_model_enabled:
            ctx.chart_model = self.chart_model
            ctx.latest_image = lambda: (f.image if (f := self.latest_frame()) is not None else None)
        return ctx

    # ------------------------------------------------------------ browser capture (hosted app)
    def latest_frame(self) -> Frame | None:
        """Newest frame from either source: local capture (mss) or the visitor's browser tab share."""
        a, b = self.capture.latest(), self.browser_frame
        if a is None or b is None:
            return a or b
        return a if a.captured_at >= b.captured_at else b

    def ingest_browser_frame(self, image: Any) -> ScreenAnalysis | None:
        """A frame shared from the visitor's browser (getDisplayMedia). Unchanged frames are skipped;
        changed ones go through the same OCR + chart-reader pipeline as local capture."""
        import time as _t

        import numpy as np

        from capture.screen import Region
        from capture.worker import fingerprint

        fp = fingerprint(image)
        prev = self.browser_frame
        if prev is not None and self.settings.dedup_threshold > 0 and float(np.abs(fp - prev.fingerprint).max()) < self.settings.dedup_threshold:
            return None
        self._browser_seq += 1
        frame = Frame(1_000_000 + self._browser_seq, utcnow(), _t.monotonic(), image, Region(0, 0, *image.size), "browser tab", fp)
        self.browser_frame = frame
        return self.extraction.process_now(frame)

    def shutdown(self) -> None:
        self.stop_monitoring()


def build_runtime(settings: Settings, source_factory: type[ScreenSource] | Any = MSSScreenSource, router: ModelRouter | None = None) -> Runtime:
    return Runtime(settings, source_factory, router=router)
