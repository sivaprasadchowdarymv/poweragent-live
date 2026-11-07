"""Background capture worker.

Design points (each covered by tests):

* One worker per process; ``CaptureWorker.start`` is idempotent, so Streamlit
  reruns can call it freely without spawning duplicate threads.
* Drift-free scheduling against ``time.monotonic`` with a configurable target
  interval; the *measured* intervals are recorded and reported, never assumed.
* A bounded ring buffer of frames. When consumers fall behind the oldest frame
  is dropped and counted; memory is therefore bounded no matter how slow
  inference is.
* Near-duplicate frames are detected with a cheap downscaled fingerprint and
  are not enqueued (counted separately from drops).
* Acquisition never waits on OCR or model inference — those run in a separate
  consumer (``extraction.ExtractionWorker``) that pulls the newest frame.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image

from core.models import utcnow
from capture.screen import CaptureError, CaptureTarget, Region, ScreenSource, resolve_target

log = logging.getLogger(__name__)

SourceFactory = Callable[[], ScreenSource]


@dataclass
class Frame:
    seq: int
    captured_at: datetime  # wall clock, UTC
    monotonic: float
    image: Image.Image
    region: Region
    target: str
    fingerprint: np.ndarray = field(repr=False)


@dataclass
class CaptureMetrics:
    captures: int = 0
    enqueued: int = 0
    duplicates: int = 0
    dropped: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    last_capture_at: datetime | None = None
    grab_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    intervals_s: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def summary(self) -> dict[str, object]:
        iv = np.array(self.intervals_s) if self.intervals_s else np.array([])
        gm = np.array(self.grab_ms) if self.grab_ms else np.array([])
        return {
            "captures": self.captures,
            "enqueued": self.enqueued,
            "duplicates_skipped": self.duplicates,
            "dropped_frames": self.dropped,
            "errors": self.errors,
            "last_error": self.last_error,
            "last_capture_at": self.last_capture_at.isoformat() if self.last_capture_at else None,
            "interval_mean_s": float(iv.mean()) if iv.size else None,
            "interval_p95_s": float(np.percentile(iv, 95)) if iv.size else None,
            "interval_max_s": float(iv.max()) if iv.size else None,
            "grab_mean_ms": float(gm.mean()) if gm.size else None,
            "grab_p95_ms": float(np.percentile(gm, 95)) if gm.size else None,
        }


def fingerprint(img: Image.Image, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    """Box-averaged 160×90 grey thumbnail. Fine enough that a live chart whose
    newest point moves by a few pixels still registers as a change (a 64×36
    thumbnail averages that away completely)."""
    small = img.convert("L").resize(size, Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-cell change (0..1). Using the maximum rather than the mean
    means a small local change (a moving line tip, a new KPI digit) counts."""
    if a.shape != b.shape:
        return 1.0
    return float(np.max(np.abs(a - b)))


class CaptureWorker:
    STATES = ("stopped", "running", "paused")

    def __init__(
        self,
        source_factory: SourceFactory,
        interval_s: float = 2.0,
        queue_size: int = 4,
        dedup_threshold: float = 0.06,
        max_pixels: int = 3840 * 2160,
    ) -> None:
        self._source_factory = source_factory
        self._lock = threading.RLock()
        self._frames: deque[Frame] = deque(maxlen=queue_size)
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread: threading.Thread | None = None
        self._target = CaptureTarget()
        self._manual_requests = 0
        self._seq = 0
        self._last_fp: np.ndarray | None = None
        self._latest: Frame | None = None
        self.interval_s = interval_s
        self.dedup_threshold = dedup_threshold
        self.max_pixels = max_pixels
        self.metrics = CaptureMetrics()
        self._threads_started = 0  # for tests: proves no duplicate workers

    # ------------------------------------------------------------------ control
    @property
    def state(self) -> str:
        if self._thread is None or not self._thread.is_alive():
            return "stopped"
        return "running" if self._resume.is_set() else "paused"

    @property
    def threads_started(self) -> int:
        return self._threads_started

    def configure(self, target: CaptureTarget | None = None, interval_s: float | None = None) -> None:
        with self._lock:
            if target is not None and target != self._target:
                self._target = target
                self._last_fp = None  # new target: first frame is never a "duplicate"
            if interval_s is not None:
                if interval_s < 0.25:
                    raise ValueError("Capture interval must be at least 0.25 s.")
                self.interval_s = float(interval_s)

    @property
    def target(self) -> CaptureTarget:
        return self._target

    def start(self) -> bool:
        """Start the worker. Returns False (and does nothing) if already alive."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._resume.set()
                return False
            self._stop.clear()
            self._resume.set()
            self._thread = threading.Thread(target=self._run, name="lda-capture", daemon=True)
            self._threads_started += 1
            self._thread.start()
            return True

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._resume.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
        with self._lock:
            self._thread = None

    def request_capture(self) -> None:
        """Ask for an immediate capture (works while running or paused)."""
        with self._lock:
            self._manual_requests += 1

    def capture_once(self) -> Frame:
        """Synchronous one-off capture used when the worker is stopped (manual button).
        Creates a short-lived source in the calling thread."""
        src = self._source_factory()
        try:
            frame = self._grab(src, force=True)
        finally:
            src.close()
        if frame is None:  # pragma: no cover - force=True never dedups
            raise CaptureError("Capture produced no frame.")
        return frame

    # ------------------------------------------------------------------ consumers
    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def take_newest(self, after_seq: int = -1) -> Frame | None:
        """Return the newest frame with seq > after_seq and clear older ones."""
        with self._lock:
            if not self._frames or self._frames[-1].seq <= after_seq:
                return None
            newest = self._frames[-1]
            self._frames.clear()
            return newest

    def wait_for_frame(self, after_seq: int, timeout: float) -> Frame | None:
        with self._new_frame:
            self._new_frame.wait_for(lambda: bool(self._frames) and self._frames[-1].seq > after_seq, timeout)
        return self.take_newest(after_seq)

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._frames)

    # ------------------------------------------------------------------ internals
    def _grab(self, src: ScreenSource, force: bool = False) -> Frame | None:
        with self._lock:
            target = self._target
        region = resolve_target(target, src)
        if region.width * region.height > self.max_pixels:
            raise CaptureError(
                f"Region {region.width}×{region.height} exceeds the {self.max_pixels:,}-pixel safety limit. Select a smaller region."
            )
        t0 = time.perf_counter()
        img = src.grab(region)
        grab_ms = (time.perf_counter() - t0) * 1000
        fp = fingerprint(img)
        now_mono = time.monotonic()
        with self._lock:
            m = self.metrics
            m.captures += 1
            m.grab_ms.append(grab_ms)
            m.consecutive_errors = 0
            m.last_capture_at = utcnow()
            if not force and self._last_fp is not None and frame_difference(fp, self._last_fp) < self.dedup_threshold:
                m.duplicates += 1
                return None
            self._seq += 1
            frame = Frame(self._seq, m.last_capture_at, now_mono, img, region, target.describe(), fp)
            self._last_fp = fp
            if len(self._frames) == self._frames.maxlen:
                m.dropped += 1  # oldest frame falls off the ring
            self._frames.append(frame)
            self._latest = frame
            m.enqueued += 1
            self._new_frame.notify_all()
            return frame

    def _run(self) -> None:
        try:
            src = self._source_factory()
        except Exception as exc:
            with self._lock:
                self.metrics.errors += 1
                self.metrics.last_error = str(exc)
            log.error("Capture source failed to open: %s", exc)
            return
        next_due = time.monotonic()
        last_tick: float | None = None
        try:
            while not self._stop.is_set():
                if not self._resume.is_set():
                    with self._lock:
                        manual = self._manual_requests
                    if manual == 0:
                        self._resume.wait(0.2)
                        next_due = time.monotonic()
                        last_tick = None
                        continue
                now = time.monotonic()
                with self._lock:
                    manual = self._manual_requests
                    self._manual_requests = 0
                if manual == 0 and now < next_due:
                    self._stop.wait(min(next_due - now, 0.1))
                    continue
                if last_tick is not None and manual == 0:
                    self.metrics.intervals_s.append(now - last_tick)
                last_tick = now
                try:
                    self._grab(src, force=manual > 0)
                except Exception as exc:
                    with self._lock:
                        self.metrics.errors += 1
                        self.metrics.consecutive_errors += 1
                        self.metrics.last_error = str(exc)
                        backoff = min(self.interval_s * (2 ** min(self.metrics.consecutive_errors, 5)), 30.0)
                    log.warning("Capture error (%s); retrying in %.1fs", exc, backoff)
                    next_due = time.monotonic() + backoff
                    continue
                # Drift-free: schedule from the previous due time, but never "catch up" in a burst.
                next_due = max(next_due + self.interval_s, time.monotonic() + 0.05)
        finally:
            src.close()
