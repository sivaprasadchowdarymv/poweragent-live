"""Extraction worker: turns captured frames into estimates, off the capture thread.

* Pulls only the newest frame (older unprocessed frames are skipped, which keeps
  latency and memory bounded when OCR is slower than capture).
* Adaptive scheduling: the effective interval grows when extraction is slow.
* The vision model runs in its own single-slot executor; if it is still busy
  with an earlier frame the new frame is not queued for it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from capture.worker import CaptureWorker, Frame
from vision.chart_reader import ScreenAnalysis, analyze_chart_image
from vision.ocr import OCRUnavailable, read_line, run_ocr

log = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    processed: int = 0
    skipped_frames: int = 0
    errors: int = 0
    last_error: str = ""
    last_frame_seq: int = -1
    ocr_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    chart_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    queue_delay_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    vision_ms: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    vision_calls: int = 0
    vision_skipped_busy: int = 0
    vision_errors: int = 0
    vision_last_error: str = ""
    chart_model_ms: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    chart_model_calls: int = 0
    chart_model_skipped_busy: int = 0
    chart_model_errors: int = 0
    chart_model_last_error: str = ""
    effective_interval_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        def stat(d: deque[float]) -> dict[str, float | None]:
            a = np.array(d) if d else None
            return {"mean": float(a.mean()) if a is not None else None, "p95": float(np.percentile(a, 95)) if a is not None else None}

        return {
            "processed_frames": self.processed,
            "skipped_frames": self.skipped_frames,
            "errors": self.errors,
            "last_error": self.last_error,
            "last_frame_seq": self.last_frame_seq,
            "ocr_ms": stat(self.ocr_ms),
            "chart_ms": stat(self.chart_ms),
            "capture_to_extraction_delay_ms": stat(self.queue_delay_ms),
            "vision_ms": stat(self.vision_ms),
            "vision_calls": self.vision_calls,
            "vision_skipped_busy": self.vision_skipped_busy,
            "vision_errors": self.vision_errors,
            "vision_last_error": self.vision_last_error,
            "chart_model_ms": stat(self.chart_model_ms),
            "chart_model_calls": self.chart_model_calls,
            "chart_model_skipped_busy": self.chart_model_skipped_busy,
            "chart_model_errors": self.chart_model_errors,
            "chart_model_last_error": self.chart_model_last_error,
            "effective_interval_s": self.effective_interval_s,
        }


def analyze_frame(frame: Frame) -> ScreenAnalysis:
    t0 = time.perf_counter()
    warnings: list[str] = []
    try:
        words = run_ocr(frame.image)
    except OCRUnavailable as exc:
        words = []
        warnings.append(f"OCR unavailable: {exc}")
    t1 = time.perf_counter()
    charts, kpis, w2 = analyze_chart_image(frame.image, words, frame.captured_at, line_reader=lambda box: read_line(frame.image, box))
    t2 = time.perf_counter()
    from storage.db import redact
    from vision.ocr import group_lines

    lines = [redact(" ".join(w.text for w in line))[:120] for line in group_lines(words)][:60]
    return ScreenAnalysis(
        captured_at=frame.captured_at,
        frame_seq=frame.seq,
        size=frame.image.size,
        charts=charts,
        kpis=kpis,
        text_lines=lines,
        ocr_ms=(t1 - t0) * 1000,
        chart_ms=(t2 - t1) * 1000,
        warnings=warnings + w2,
    )


class ExtractionWorker:
    def __init__(
        self,
        capture: CaptureWorker,
        on_result: Callable[[ScreenAnalysis], None],
        interval_s: float = 4.0,
        vision_call: Callable[[Frame], dict[str, Any]] | None = None,
        on_vision: Callable[[dict[str, Any], Frame], None] | None = None,
        vision_interval_s: float = 30.0,
        analyzer: Callable[[Frame], ScreenAnalysis] = analyze_frame,
        on_frame: Callable[[Frame, ScreenAnalysis], None] | None = None,
        chart_model_call: Callable[[Frame, ScreenAnalysis], Any] | None = None,
        on_chart_model: Callable[[Any, Frame, ScreenAnalysis], None] | None = None,
        chart_model_interval_s: float = 30.0,
    ) -> None:
        self.on_frame = on_frame
        self.chart_model_call = chart_model_call
        self.on_chart_model = on_chart_model
        self.chart_model_interval_s = chart_model_interval_s
        self.chart_model_enabled = False
        self._chart_model_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lda-chart_model")
        self._chart_model_future: Future[None] | None = None
        self._last_chart_model = 0.0
        self.capture = capture
        self.on_result = on_result
        self.interval_s = interval_s
        self.vision_call = vision_call
        self.on_vision = on_vision
        self.vision_interval_s = vision_interval_s
        self.vision_enabled = False
        self.ocr_enabled = True
        self.analyzer = analyzer
        self.metrics = ExtractionMetrics(effective_interval_s=interval_s)
        self.status = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._vision_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lda-vision")
        self._vision_future: Future[None] | None = None
        self._last_vision = 0.0
        self.threads_started = 0

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.alive:
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="lda-extract", daemon=True)
            self.threads_started += 1
            self._thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self._thread = None
        self.status = "idle"

    @property
    def vision_busy(self) -> bool:
        return self._vision_future is not None and not self._vision_future.done()

    @property
    def chart_model_busy(self) -> bool:
        return self._chart_model_future is not None and not self._chart_model_future.done()

    def process_now(self, frame: Frame) -> ScreenAnalysis:
        """Synchronous extraction for a manual capture."""
        return self._process(frame)

    def _process(self, frame: Frame) -> ScreenAnalysis:
        self.status = "extracting"
        delay = (time.monotonic() - frame.monotonic) * 1000
        try:
            analysis = self.analyzer(frame) if self.ocr_enabled else ScreenAnalysis(frame.captured_at, frame.seq, frame.image.size, [], [], [], 0, 0, ["OCR/chart extraction is turned off."])
        except Exception as exc:
            self.metrics.errors += 1
            self.metrics.last_error = str(exc)
            self.status = "error"
            raise
        m = self.metrics
        m.processed += 1
        m.last_frame_seq = frame.seq
        m.ocr_ms.append(analysis.ocr_ms)
        m.chart_ms.append(analysis.chart_ms)
        m.queue_delay_ms.append(delay)
        self.on_result(analysis)
        if self.on_frame is not None:
            try:
                self.on_frame(frame, analysis)
            except Exception as exc:  # optional hook must never break extraction
                log.warning("on_frame hook failed: %s", exc)
        self._maybe_vision(frame)
        self._maybe_chart_model(frame, analysis)
        self.status = "idle"
        return analysis

    def _maybe_vision(self, frame: Frame) -> None:
        if not (self.vision_enabled and self.vision_call and self.on_vision):
            return
        if time.monotonic() - self._last_vision < self.vision_interval_s:
            return
        if self.vision_busy:
            self.metrics.vision_skipped_busy += 1
            return
        self._last_vision = time.monotonic()
        self._vision_future = self._vision_pool.submit(self._vision_job, frame)

    def _vision_job(self, frame: Frame) -> None:
        t0 = time.perf_counter()
        try:
            assert self.vision_call is not None and self.on_vision is not None
            notes = self.vision_call(frame)
            self.metrics.vision_calls += 1
            self.metrics.vision_ms.append((time.perf_counter() - t0) * 1000)
            self.on_vision(notes, frame)
        except Exception as exc:
            self.metrics.vision_errors += 1
            self.metrics.vision_last_error = str(exc)
            log.warning("Vision model call failed: %s", exc)

    def _maybe_chart_model(self, frame: Frame, analysis: ScreenAnalysis, force: bool = False) -> bool:
        """Chart-model reading runs in its own single slot at its own (slow) pace:
        if the previous reading is still running, this frame is not queued."""
        if not (self.chart_model_enabled and self.chart_model_call and self.on_chart_model):
            return False
        if not force and time.monotonic() - self._last_chart_model < self.chart_model_interval_s:
            return False
        if self.chart_model_busy:
            self.metrics.chart_model_skipped_busy += 1
            return False
        self._last_chart_model = time.monotonic()
        self._chart_model_future = self._chart_model_pool.submit(self._chart_model_job, frame, analysis)
        return True

    def request_chart_model(self, frame: Frame, analysis: ScreenAnalysis) -> bool:
        """Run a chart-model reading now (manual button), unless one is already running."""
        return self._maybe_chart_model(frame, analysis, force=True)

    def _chart_model_job(self, frame: Frame, analysis: ScreenAnalysis) -> None:
        t0 = time.perf_counter()
        try:
            assert self.chart_model_call is not None and self.on_chart_model is not None
            readings = self.chart_model_call(frame, analysis)
            self.metrics.chart_model_calls += 1
            self.metrics.chart_model_ms.append((time.perf_counter() - t0) * 1000)
            errs = [r.error for r in readings if getattr(r, "error", "")]
            if errs:
                self.metrics.chart_model_errors += 1
                self.metrics.chart_model_last_error = errs[-1]
            self.on_chart_model(readings, frame, analysis)
        except Exception as exc:
            self.metrics.chart_model_errors += 1
            self.metrics.chart_model_last_error = str(exc)
            log.warning("chart-model reading failed: %s", exc)

    def _run(self) -> None:
        last_seq = -1
        next_due = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_due:
                self._stop.wait(min(next_due - now, 0.25))
                continue
            frame = self.capture.wait_for_frame(last_seq, timeout=0.5)
            if frame is None:
                continue
            if last_seq >= 0 and frame.seq > last_seq + 1:
                self.metrics.skipped_frames += frame.seq - last_seq - 1
            last_seq = frame.seq
            t0 = time.monotonic()
            try:
                self._process(frame)
            except Exception as exc:  # keep the worker alive
                log.exception("Extraction failed: %s", exc)
            took = time.monotonic() - t0
            eff = max(self.interval_s, 1.25 * took)
            self.metrics.effective_interval_s = eff
            next_due = t0 + eff
