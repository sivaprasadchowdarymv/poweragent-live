"""Capture worker lifecycle, bounded queue, dedup, errors, independence from inference."""

import threading
import time

import numpy as np
import pytest
from PIL import Image

from capture.screen import CaptureError, CaptureTarget, Region, resolve_target
from capture.worker import CaptureWorker
from vision.chart_reader import ScreenAnalysis
from vision.extraction import ExtractionWorker

from tests.conftest import FakeScreen


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_start_is_idempotent_no_duplicate_threads():
    w = CaptureWorker(FakeScreen, interval_s=0.25)
    try:
        assert w.start() is True
        for _ in range(20):  # simulates many Streamlit reruns
            assert w.start() is False
        assert w.threads_started == 1
        assert len([t for t in threading.enumerate() if t.name == "lda-capture"]) == 1
    finally:
        w.stop()
    assert w.state == "stopped"
    assert not [t for t in threading.enumerate() if t.name == "lda-capture" and t.is_alive()]


def test_queue_is_bounded_and_drops_are_counted():
    w = CaptureWorker(FakeScreen, interval_s=0.25, queue_size=3, dedup_threshold=0.0)
    w.start()
    try:
        assert wait_until(lambda: w.metrics.captures >= 8, 6)
        assert w.queue_depth() <= 3
        assert w.metrics.dropped >= w.metrics.enqueued - 3 - 1
    finally:
        w.stop()


def test_duplicate_frames_are_skipped():
    still = Image.new("RGB", (320, 200), "white")
    w = CaptureWorker(lambda: FakeScreen(provider=lambda n: still), interval_s=0.25)
    w.start()
    try:
        assert wait_until(lambda: w.metrics.captures >= 4, 5)
        assert w.metrics.enqueued == 1
        assert w.metrics.duplicates >= 3
    finally:
        w.stop()


def test_small_local_change_is_not_a_duplicate():
    """A chart whose newest point moves a few pixels must still count as new."""
    from PIL import ImageDraw

    def frame(n):
        img = Image.new("RGB", (800, 500), "white")
        d = ImageDraw.Draw(img)
        d.line([(50, 400), (700, 300), (750, 300 - 4 * n)], fill=(42, 120, 214), width=2)
        return img

    w = CaptureWorker(lambda: FakeScreen(provider=frame, size=(800, 500)), interval_s=0.25)
    w.start()
    try:
        assert wait_until(lambda: w.metrics.captures >= 4, 5)
        assert w.metrics.duplicates == 0 and w.metrics.enqueued == w.metrics.captures
    finally:
        w.stop()


def test_pause_resume_and_manual_capture_while_paused():
    w = CaptureWorker(FakeScreen, interval_s=0.25, dedup_threshold=0.0)
    w.start()
    try:
        assert wait_until(lambda: w.metrics.captures >= 2)
        w.pause()
        time.sleep(0.4)
        n = w.metrics.captures
        time.sleep(0.6)
        assert w.metrics.captures == n and w.state == "paused"
        w.request_capture()
        assert wait_until(lambda: w.metrics.captures == n + 1, 3)
        w.resume()
        assert wait_until(lambda: w.metrics.captures >= n + 3)
    finally:
        w.stop()


def test_capture_errors_do_not_crash_worker():
    w = CaptureWorker(lambda: FakeScreen(fail=True), interval_s=0.25)
    w.start()
    try:
        assert wait_until(lambda: w.metrics.errors >= 2, 6)
        assert w.state == "running"
        assert "simulated" in w.metrics.last_error
    finally:
        w.stop()


def test_interval_accuracy():
    w = CaptureWorker(FakeScreen, interval_s=0.3, dedup_threshold=0.0)
    w.start()
    try:
        assert wait_until(lambda: len(w.metrics.intervals_s) >= 8, 8)
    finally:
        w.stop()
    iv = np.array(w.metrics.intervals_s)
    assert abs(iv.mean() - 0.3) < 0.03
    assert iv.max() < 0.45


def test_capture_continues_during_slow_extraction():
    """Slow inference must not block acquisition, and memory stays bounded."""
    w = CaptureWorker(FakeScreen, interval_s=0.25, queue_size=2, dedup_threshold=0.0)
    seen = []

    def slow(frame):
        time.sleep(1.2)
        return ScreenAnalysis(frame.captured_at, frame.seq, frame.image.size, [], [], [], 0, 0)

    ex = ExtractionWorker(w, on_result=seen.append, interval_s=0.5, analyzer=slow)
    w.start()
    ex.start()
    try:
        assert wait_until(lambda: len(seen) >= 2, 8)
        caps_at_two = w.metrics.captures
        assert caps_at_two >= 7  # ~2.4 s of extraction -> capture kept going
        assert w.queue_depth() <= 2
        assert ex.metrics.skipped_frames > 0  # only the newest frame is processed
        assert ex.metrics.effective_interval_s >= 1.2  # adaptive scheduling backed off
        assert ex.start() is False and ex.threads_started == 1
    finally:
        ex.stop()
        w.stop()


def test_resolve_target_handles_missing_monitor_and_offscreen_region():
    src = FakeScreen(size=(800, 600))
    with pytest.raises(CaptureError, match="Monitor 3"):
        resolve_target(CaptureTarget("monitor", monitor_index=3), src)
    clipped = resolve_target(CaptureTarget("region", region=Region(700, 500, 400, 400)), src)
    assert clipped == Region(700, 500, 100, 100)
    with pytest.raises(CaptureError, match="outside"):
        resolve_target(CaptureTarget("region", region=Region(5000, 5000, 10, 10)), src)


def test_capture_once_when_stopped():
    w = CaptureWorker(FakeScreen)
    f = w.capture_once()
    assert f.seq == 1 and f.image.size == (320, 200)
    assert w.state == "stopped"
