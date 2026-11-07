"""Screen sources: monitors, regions and (on Windows) application windows.

``ScreenSource`` is a small protocol so tests can inject a fake source and the
capture worker never touches hardware directly.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """Capture failed; message is user-facing and actionable."""


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Region:
        try:
            r = cls(int(d["left"]), int(d["top"]), int(d["width"]), int(d["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError("A capture region needs integer left, top, width and height.") from exc
        if r.width < 1 or r.height < 1:
            raise CaptureError("Capture width and height must be positive.")
        return r

    def clip_to(self, bounds: Region) -> Region | None:
        left = max(self.left, bounds.left)
        top = max(self.top, bounds.top)
        right = min(self.left + self.width, bounds.left + bounds.width)
        bottom = min(self.top + self.height, bounds.top + bounds.height)
        if right <= left or bottom <= top:
            return None
        return Region(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class CaptureTarget:
    """What to capture. ``kind`` is 'monitor', 'region' or 'window'."""

    kind: str = "monitor"
    monitor_index: int = 1  # MSS numbering: 1 = first physical monitor
    region: Region | None = None
    window_title: str | None = None

    def describe(self) -> str:
        if self.kind == "window":
            return f"window '{self.window_title}'"
        if self.kind == "region" and self.region:
            r = self.region
            return f"region {r.width}×{r.height} at ({r.left},{r.top})"
        return f"monitor {self.monitor_index}"


class ScreenSource(Protocol):
    def monitors(self) -> list[Region]: ...

    def grab(self, region: Region) -> Image.Image: ...

    def close(self) -> None: ...


def _mss_factory() -> Any:
    import mss

    cls = getattr(mss, "MSS", None) or mss.mss
    return cls()


class MSSScreenSource:
    """MSS-backed source. MSS handles are thread-affine on Windows, so create
    this object inside the thread that uses it (the capture worker does)."""

    def __init__(self) -> None:
        try:
            self._sct = _mss_factory()
        except Exception as exc:  # pragma: no cover - depends on display
            raise CaptureError(
                "Could not open the screen for capture. Run the app inside your logged-in desktop session; "
                "over Remote Desktop keep the session window open; on macOS grant Screen Recording permission. "
                f"Details: {exc}"
            ) from exc

    def monitors(self) -> list[Region]:
        mons = self._sct.monitors[1:] or self._sct.monitors[:1]
        return [Region(int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"])) for m in mons]

    def grab(self, region: Region) -> Image.Image:
        try:
            shot = self._sct.grab(region.as_dict())
        except Exception as exc:
            raise CaptureError(f"Screen grab failed for {region}: {exc}") from exc
        return Image.frombytes("RGB", shot.size, shot.rgb)

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:  # pragma: no cover
            pass


def list_monitors() -> list[Region]:
    """Monitors visible right now; empty list if no display is reachable."""
    try:
        src = MSSScreenSource()
    except CaptureError as exc:
        log.warning("%s", exc)
        return []
    try:
        return src.monitors()
    finally:
        src.close()


# --- Windows application-window support (optional) -------------------------


@dataclass(frozen=True)
class WindowInfo:
    title: str
    region: Region
    minimized: bool


def list_windows() -> list[WindowInfo]:
    """Visible top-level windows. Uses ``pygetwindow`` when installed (Windows);
    returns an empty list elsewhere so the UI falls back to monitor/region."""
    if sys.platform != "win32":
        return []
    try:
        import pygetwindow as gw  # type: ignore[import-not-found]
    except ImportError:
        return []
    out: list[WindowInfo] = []
    for w in gw.getAllWindows():
        title = (w.title or "").strip()
        if not title or w.width < 50 or w.height < 50:
            continue
        out.append(WindowInfo(title, Region(int(w.left), int(w.top), int(w.width), int(w.height)), bool(w.isMinimized)))
    return out


def resolve_target(target: CaptureTarget, source: ScreenSource) -> Region:
    """Turn a target into a concrete on-screen region, handling missing monitors,
    resolution changes and minimized windows."""
    mons = source.monitors()
    if not mons:
        raise CaptureError("No monitors detected. Check that a display is connected and the session is unlocked.")
    if target.kind == "window":
        matches = [w for w in list_windows() if w.title == target.window_title]
        if not matches:
            raise CaptureError(f"Window '{target.window_title}' is not open any more. Pick another target.")
        win = matches[0]
        if win.minimized:
            raise CaptureError(f"Window '{win.title}' is minimized. Restore it to resume capture.")
        region = win.region
    elif target.kind == "region" and target.region is not None:
        region = target.region
    else:
        idx = target.monitor_index
        if idx < 1 or idx > len(mons):
            raise CaptureError(f"Monitor {idx} is not available (found {len(mons)}). Select another monitor.")
        return mons[idx - 1]
    # Clip region/window to the union of monitors so a resolution change never crashes capture.
    left = min(m.left for m in mons)
    top = min(m.top for m in mons)
    right = max(m.left + m.width for m in mons)
    bottom = max(m.top + m.height for m in mons)
    clipped = region.clip_to(Region(left, top, right - left, bottom - top))
    if clipped is None:
        raise CaptureError("The selected region is outside every monitor (did the resolution change?). Reselect it.")
    return clipped
