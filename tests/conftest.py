from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.screen import Region  # noqa: E402
from storage.db import Database  # noqa: E402


class FakeScreen:
    """Deterministic screen source: each grab returns the next image from a
    provider; can be told to fail. No hardware involved."""

    instances = 0

    def __init__(self, provider=None, fail=False, size=(320, 200)):
        FakeScreen.instances += 1
        self.size = size
        self.provider = provider
        self.fail = fail
        self.grabs = 0
        self.lock = threading.Lock()

    def monitors(self):
        return [Region(0, 0, *self.size)]

    def grab(self, region):
        with self.lock:
            self.grabs += 1
            n = self.grabs
        if self.fail:
            from capture.screen import CaptureError

            raise CaptureError("simulated capture failure")
        if self.provider is not None:
            return self.provider(n).resize((region.width, region.height))
        rng = np.random.default_rng(n)
        return Image.fromarray(rng.integers(0, 255, (region.height, region.width, 3), dtype=np.uint8))

    def close(self):
        pass


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture
def ts_index():
    def make(n=100, freq="2s", start="2026-10-29 10:00:00"):
        return pd.date_range(start, periods=n, freq=freq, tz="UTC")

    return make
