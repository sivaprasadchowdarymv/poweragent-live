"""Local OCR via Tesseract (pytesseract). Degrades gracefully when absent."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"^[−\-+]?(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?([eE][−\-+]?\d+)?$")
SUFFIX_MULT = {"k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "µ": 1e-6, "u": 1e-6}
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


class OCRUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrWord:
    text: str
    conf: float  # 0..1
    left: int
    top: int
    width: int
    height: int
    line_key: tuple[int, int, int] = (0, 0, 0)

    @property
    def cx(self) -> float:
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        return self.top + self.height / 2

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@lru_cache(maxsize=1)
def ocr_status() -> tuple[bool, str]:
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract is not installed (pip install pytesseract)."
    import os
    import sys

    if sys.platform == "win32" and not shutil.which("tesseract"):
        for cand in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(cand) and not str(pytesseract.pytesseract.tesseract_cmd).lower().endswith(".exe"):
                pytesseract.pytesseract.tesseract_cmd = cand
    if not shutil.which("tesseract") and not str(getattr(pytesseract.pytesseract, "tesseract_cmd", "")).lower().endswith(".exe"):
        return False, (
            "The Tesseract program was not found. On Windows install it from the UB-Mannheim build and either add "
            "it to PATH or set TESSERACT_CMD to tesseract.exe."
        )
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:
        return False, f"Tesseract could not start: {exc}"
    return True, f"Tesseract {version}"


def configure_tesseract(cmd: str | None) -> None:
    if not cmd:
        return
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd
        ocr_status.cache_clear()
    except ImportError:
        pass


def run_ocr(img: Image.Image, min_conf: float = 0.3, upscale_below: int = 1600) -> list[OcrWord]:
    ok, msg = ocr_status()
    if not ok:
        raise OCRUnavailable(msg)
    import pytesseract

    scale = 2 if max(img.size) < upscale_below else 1
    work = img.convert("L")
    # Dark-theme dashboards: Tesseract prefers dark text on light background.
    if float(np.asarray(work.resize((32, 32))).mean()) < 110:
        work = ImageOps.invert(work)
    work = Image.fromarray(remove_rules(np.asarray(work)))
    if scale > 1:
        work = work.resize((work.width * scale, work.height * scale), Image.Resampling.LANCZOS)
    data = pytesseract.image_to_data(work, config="--psm 11", output_type=pytesseract.Output.DICT)
    words: list[OcrWord] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        try:
            conf = float(data["conf"][i]) / 100.0
        except (TypeError, ValueError):
            conf = 0.0
        if not text or conf < min_conf:
            continue
        words.append(
            OcrWord(
                text=text,
                conf=conf,
                left=int(data["left"][i] / scale),
                top=int(data["top"][i] / scale),
                width=max(1, int(data["width"][i] / scale)),
                height=max(1, int(data["height"][i] / scale)),
                line_key=(int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i])),
            )
        )
    return words


def remove_rules(gray: np.ndarray, ink_level: int = 160) -> np.ndarray:
    """Erase long horizontal/vertical lines (axes, spines, gridlines) and the short
    tick marks attached to them, so OCR does not read a tick as an extra digit
    (e.g. '20' + tick -> '207'). Text strokes are far shorter than these rules."""
    from scipy import ndimage

    g = gray.copy()
    H, W = g.shape
    ink = g < ink_level
    min_h = max(40, int(0.2 * W))
    min_v = max(40, int(0.2 * H))
    rules = np.zeros_like(ink)
    # Horizontal runs
    for axis, min_len in ((1, min_h), (0, min_v)):
        arr = ink if axis == 1 else ink.T
        out = rules if axis == 1 else rules.T
        for i in range(arr.shape[0]):
            row = arr[i]
            if row.sum() < min_len:
                continue
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            for s, e in zip(starts, ends, strict=False):
                if e - s >= min_len:
                    out[i, s:e] = True
    if not rules.any():
        return g
    near_rule = ndimage.binary_dilation(rules, iterations=2)
    remaining = ink & ~rules
    labels, n = ndimage.label(remaining)
    if n:
        slices = ndimage.find_objects(labels)
        for idx, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            # tick marks: tiny thin segments touching a rule
            if (h <= 3 and 2 <= w <= 14) or (w <= 3 and 2 <= h <= 14):
                comp = labels[sl] == idx
                if (comp & near_rule[sl]).any():
                    rules[sl] |= comp
    g[rules] = 255
    return g


def read_line(img: Image.Image, box: tuple[int, int, int, int]) -> str:
    """OCR a single text line inside ``box`` (left, top, right, bottom)."""
    ok, _ = ocr_status()
    if not ok:
        return ""
    import pytesseract

    crop = img.crop(box).convert("L")
    if crop.width < 4 or crop.height < 4:
        return ""
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)
    try:
        text = pytesseract.image_to_string(crop, config="--psm 7").strip()
    except Exception:
        return ""
    return re.sub(r"[^\w .%()/+-]", "", text).strip()[:60]


def parse_number(text: str) -> float | None:
    """Parse tick-style numbers: '1,200', '-3.5', '2k', '1e3', '45%', '230V'. Returns None if not numeric."""
    t = text.strip().replace("−", "-").replace("O", "0") if any(c.isdigit() for c in text) else text.strip()
    t = re.sub(r"(%|[A-Za-zΩ°]{1,3})$", lambda m: m.group(0) if m.group(0) in SUFFIX_MULT else "", t)
    mult = 1.0
    if t and t[-1] in SUFFIX_MULT and len(t) > 1 and (t[-2].isdigit() or t[-2] == "."):
        mult = SUFFIX_MULT[t[-1]]
        t = t[:-1]
    if not t or not NUMBER_RE.match(t) or not any(c.isdigit() for c in t):
        return None
    try:
        return float(t.replace(",", "")) * mult
    except ValueError:
        return None


def parse_time_of_day(text: str) -> float | None:
    """'10:05' or '10:05:30' -> seconds since midnight."""
    m = TIME_RE.match(text.strip())
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if h > 23 or mi > 59 or s > 59:
        return None
    return float(h * 3600 + mi * 60 + s)


def group_lines(words: list[OcrWord], y_tol: float = 0.6) -> list[list[OcrWord]]:
    """Group words into visual lines by vertical overlap, left-to-right."""
    lines: list[list[OcrWord]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.left)):
        for line in lines:
            ref = line[0]
            if abs(w.cy - ref.cy) <= y_tol * max(ref.height, w.height):
                line.append(w)
                break
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.left) for line in lines]
