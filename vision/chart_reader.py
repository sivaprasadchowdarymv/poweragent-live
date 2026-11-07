"""Chart and KPI extraction from screenshots.

Pipeline
--------
1. OCR the frame (Tesseract) and parse numeric / time-of-day tokens.
2. Cluster tick labels into axes: vertically stacked, right-aligned numbers
   form a y-axis; a horizontal row of numbers/times forms an x-axis.
3. Calibrate each axis with a RANSAC fit of pixel→value for linear, log10 and
   time scales. Reversed axes fall out of the sign of the slope. OCR misreads
   become outliers instead of corrupting the fit.
4. Inside each plot area, segment saturated colours into series, decide
   line/area/bar, and map pixels to data values through the calibration.
5. Read KPI tiles ("Voltage 231.4 V") from the remaining numbers.

Everything produced here is an *estimate* with a confidence score; it is
never labelled as verified data. Grey/black series are not separated from
gridlines and are reported as a limitation.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from PIL import Image

from vision.ocr import OcrWord, group_lines, parse_number, parse_time_of_day

UNIT_TOKENS = {
    "v", "kv", "mv", "a", "ma", "ka", "w", "kw", "mw", "gw", "kwh", "mwh", "wh", "va", "kva", "var", "kvar",
    "hz", "%", "°c", "c", "°f", "pf", "rpm", "bar", "psi", "pa", "kpa", "s", "ms", "°", "ω", "ohm",
}


# ------------------------------------------------------------------ data types
@dataclass
class AxisCalibration:
    axis: str  # 'x' or 'y'
    scale: str  # 'linear' | 'log' | 'time'
    a: float
    b: float
    reversed: bool
    ticks: list[tuple[float, float]]
    inliers: int
    residual: float
    confidence: float

    def to_value(self, p: float | np.ndarray) -> Any:
        v = self.a + self.b * np.asarray(p, dtype=float)
        return np.power(10.0, v) if self.scale == "log" else v

    def to_pixel(self, value: float) -> float:
        v = math.log10(value) if self.scale == "log" else value
        return (v - self.a) / self.b


@dataclass
class SeriesEstimate:
    name: str
    color: str
    kind: str  # line | area | bar
    x: list[float]
    y: list[float]
    x_is_relative: bool
    latest: float | None
    minimum: float | None
    maximum: float | None
    clipped: bool
    confidence: float

    def points(self) -> list[tuple[float, float]]:
        return list(zip(self.x, self.y, strict=False))


@dataclass
class ChartEstimate:
    id: str
    bbox: tuple[int, int, int, int]  # plot area: left, top, right, bottom (image pixels)
    title: str
    chart_type: str
    x_axis: AxisCalibration | None
    y_axis: AxisCalibration
    series: list[SeriesEstimate]
    unit: str
    confidence: float
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title or "(untitled chart)",
            "type": self.chart_type,
            "unit": self.unit,
            "y_axis": {
                "scale": self.y_axis.scale,
                "reversed": self.y_axis.reversed,
                "range": _axis_range(self.y_axis),
                "confidence": round(self.y_axis.confidence, 2),
            },
            "x_axis": None
            if self.x_axis is None
            else {"scale": self.x_axis.scale, "range": _axis_range(self.x_axis), "confidence": round(self.x_axis.confidence, 2)},
            "series": [
                {
                    "name": s.name,
                    "color": s.color,
                    "kind": s.kind,
                    "points": len(s.x),
                    "latest_estimate": _r(s.latest),
                    "min_estimate": _r(s.minimum),
                    "max_estimate": _r(s.maximum),
                    "clipped": s.clipped,
                    "confidence": round(s.confidence, 2),
                }
                for s in self.series
            ],
            "confidence": round(self.confidence, 2),
            "warnings": self.warnings,
            "status": "estimated",
        }


@dataclass
class KpiEstimate:
    label: str
    value: float
    unit: str
    confidence: float
    bbox: tuple[int, int, int, int]
    raw_text: str


@dataclass
class ScreenAnalysis:
    captured_at: datetime
    frame_seq: int
    size: tuple[int, int]
    charts: list[ChartEstimate]
    kpis: list[KpiEstimate]
    text_lines: list[str]
    ocr_ms: float
    chart_ms: float
    warnings: list[str] = field(default_factory=list)
    vision: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at.isoformat(),
            "frame_seq": self.frame_seq,
            "image_size": list(self.size),
            "charts": [c.summary() for c in self.charts],
            "kpis": [
                {"label": k.label, "value_estimate": _r(k.value), "unit": k.unit, "confidence": round(k.confidence, 2)}
                for k in self.kpis
            ],
            "warnings": self.warnings,
            "vision_model_notes": self.vision,
            "status": "estimated from screenshot",
        }


def _r(v: float | None, digits: int = 4) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.{digits}g}")


def _axis_range(ax: AxisCalibration) -> list[float | None]:
    vals = [v for _, v in ax.ticks]
    return [_r(min(vals)), _r(max(vals))] if vals else [None, None]


# ------------------------------------------------------------------ calibration
def fit_axis(axis: str, pixels: list[float], values: list[float], confs: list[float], scale_hint: str | None = None) -> AxisCalibration | None:
    """Robust pixel→value fit. Tries linear and log (and 'time' when hinted)."""
    if len(pixels) < 3:
        return None
    p = np.asarray(pixels, float)
    v = np.asarray(values, float)
    candidates: list[tuple[str, np.ndarray]] = []
    if scale_hint == "time":
        candidates.append(("time", v))
    else:
        candidates.append(("linear", v))
        if np.all(v > 0):
            candidates.append(("log", np.log10(v)))
    best: AxisCalibration | None = None
    best_key: tuple[float, ...] | None = None
    for scale, vv in candidates:
        span = float(vv.max() - vv.min())
        if span <= 0:
            continue
        tol = 0.02 * span
        best_inl: np.ndarray | None = None
        for i, j in itertools.combinations(range(len(p)), 2):
            if p[i] == p[j] or vv[i] == vv[j]:
                continue
            b = (vv[j] - vv[i]) / (p[j] - p[i])
            a = vv[i] - b * p[i]
            inl = np.abs(a + b * p - vv) <= tol
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl = inl
        if best_inl is None or best_inl.sum() < 3:
            continue
        b, a = np.polyfit(p[best_inl], vv[best_inl], 1)
        resid = float(np.sqrt(np.mean((a + b * p[best_inl] - vv[best_inl]) ** 2)) / span)
        n_in = int(best_inl.sum())
        # Prefer more inliers, then linear over log for ties (simpler model), then lower residual.
        key = (n_in, 1.0 if scale != "log" else 0.0, -resid)
        if best_key is None or key > best_key:
            reversed_ = bool(b > 0) if axis == "y" else bool(b < 0)
            conf_ocr = float(np.mean(np.asarray(confs)[best_inl])) if confs else 0.8
            confidence = (n_in / len(p)) * min(1.0, n_in / 4) * conf_ocr * max(0.0, 1 - 10 * resid)
            best = AxisCalibration(
                axis=axis,
                scale=scale,
                a=float(a),
                b=float(b),
                reversed=reversed_,
                ticks=[(float(pp), float(vl)) for pp, vl, ok in zip(p, values, best_inl, strict=False) if ok],
                inliers=n_in,
                residual=resid,
                confidence=float(np.clip(confidence, 0, 1)),
            )
            best_key = key
    return best


# ------------------------------------------------------------------ axis grouping
@dataclass
class _Tick:
    word: OcrWord
    value: float


def _numeric_ticks(words: list[OcrWord]) -> tuple[list[_Tick], list[_Tick]]:
    nums, times = [], []
    for w in words:
        t = parse_time_of_day(w.text)
        if t is not None:
            times.append(_Tick(w, t))
            continue
        n = parse_number(w.text)
        if n is not None:
            nums.append(_Tick(w, n))
    return nums, times


def _cluster(items: list[_Tick], key: Any, tol: float) -> list[list[_Tick]]:
    groups: list[list[_Tick]] = []
    for it in sorted(items, key=key):
        if groups and abs(key(it) - key(groups[-1][-1])) <= tol:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _y_axis_groups(ticks: list[_Tick], used: set[int]) -> list[list[_Tick]]:
    out = []
    for g in _cluster([t for t in ticks if id(t.word) not in used], key=lambda t: t.word.right, tol=8):
        # Within a right-aligned column there may be several charts stacked vertically:
        # split on large vertical gaps.
        g = sorted(g, key=lambda t: t.word.cy)
        parts: list[list[_Tick]] = [[g[0]]]
        for prev, cur in itertools.pairwise(g):
            gaps = [b.word.cy - a.word.cy for a, b in itertools.pairwise(parts[-1])] or [cur.word.cy - prev.word.cy]
            if cur.word.cy - prev.word.cy > 2.5 * float(np.median(gaps)) and len(parts[-1]) >= 3:
                parts.append([cur])
            else:
                parts[-1].append(cur)
        for part in parts:
            if len(part) >= 3 and part[-1].word.cy - part[0].word.cy >= 40 and len({t.value for t in part}) >= 3:
                out.append(part)
    return out


def _x_axis_groups(ticks: list[_Tick]) -> list[list[_Tick]]:
    out = []
    for g in _cluster(ticks, key=lambda t: t.word.cy, tol=5):
        g = sorted(g, key=lambda t: t.word.cx)
        # split on large horizontal gaps (side-by-side charts)
        parts: list[list[_Tick]] = [[g[0]]]
        for prev, cur in itertools.pairwise(g):
            gaps = [b.word.cx - a.word.cx for a, b in itertools.pairwise(parts[-1])] or [cur.word.cx - prev.word.cx]
            if cur.word.cx - prev.word.cx > 2.5 * float(np.median(gaps)) and len(parts[-1]) >= 3:
                parts.append([cur])
            else:
                parts[-1].append(cur)
        for part in parts:
            if len(part) >= 3 and part[-1].word.cx - part[0].word.cx >= 60 and len({t.value for t in part}) >= 3:
                out.append(part)
    return out


# ------------------------------------------------------------------ colour series
def _rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(-1)
    mn = arr.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def _hue_clusters(h: np.ndarray, mask: np.ndarray, min_pixels: int) -> list[tuple[float, float]]:
    hist, _ = np.histogram(h[mask], bins=72, range=(0, 360))
    active = hist >= max(3, min_pixels // 6)
    clusters: list[list[int]] = []
    for i in range(72):
        if active[i]:
            if clusters and clusters[-1][-1] == i - 1:
                clusters[-1].append(i)
            else:
                clusters.append([i])
    if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == 71:  # wrap red hue
        clusters[0] = clusters.pop() + clusters[0]
    out = []
    for c in clusters:
        if hist[c].sum() >= min_pixels:
            lo, hi = c[0] * 5.0, (c[-1] + 1) * 5.0
            out.append((lo, hi))
    return out


def _in_hue(h: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (h >= lo) & (h < hi) if lo < hi else (h >= lo) | (h < hi)


def _extract_series(
    rgb: np.ndarray,
    plot: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
    y_axis: AxisCalibration,
    x_axis: AxisCalibration | None,
    swatches: list[tuple[int, int, int, int]] | None = None,
) -> list[SeriesEstimate]:
    left, top, right, bottom = plot
    crop = rgb[top:bottom, left:right].astype(np.float32) / 255.0
    if crop.size == 0:
        return []
    h, s, v = _rgb_to_hsv(crop)
    mask = (s > 0.3) & (v > 0.25)
    for bl, bt, br, bb in text_boxes:  # ignore annotation text and legend swatches inside the plot
        x0, x1 = max(bl - left - 4 * (bb - bt) - 2, 0), min(br - left + 1, crop.shape[1])
        y0, y1 = max(bt - top - 1, 0), min(bb - top + 1, crop.shape[0])
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = False
    for bl, bt, br, bb in swatches or []:
        x0, x1 = max(bl - left, 0), min(br - left, crop.shape[1])
        y0, y1 = max(bt - top, 0), min(bb - top, crop.shape[0])
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = False
    H, W = mask.shape
    clusters = _hue_clusters(h, mask, min_pixels=max(25, int(0.0015 * H * W)))
    series: list[SeriesEstimate] = []
    for lo, hi in clusters:
        cm = mask & _in_hue(h, lo, hi)
        cols = np.where(cm.any(axis=0))[0]
        if cols.size < max(5, 0.05 * W):
            continue
        counts = cm.sum(axis=0)
        first_row = np.argmax(cm, axis=0)
        extent = np.where(counts > 0, H - first_row, 0)
        active = counts > 0
        fill_ratio = float(np.median(counts[active] / np.maximum(extent[active], 1)))
        median_extent = float(np.median(extent[active]))
        runs = _runs(active)
        if fill_ratio > 0.6 and median_extent > 8:
            gap_cols = (cols[-1] - cols[0] + 1) - cols.size
            kind = "bar" if len(runs) >= 2 and gap_cols > 0.15 * (cols[-1] - cols[0] + 1) else "area"
        else:
            kind = "line"
        if kind == "bar":
            px_x = np.array([(a + b) / 2 for a, b in runs], float)
            px_y = np.array([np.median(first_row[a : b + 1]) for a, b in runs], float)
        elif kind == "area":
            px_x = cols.astype(float)
            px_y = first_row[cols].astype(float)
        else:
            px_x = cols.astype(float)
            px_y = np.array([np.median(np.where(cm[:, c])[0]) for c in cols], float)
        if px_x.size > 300:
            idx = np.linspace(0, px_x.size - 1, 300).round().astype(int)
            px_x, px_y = px_x[idx], px_y[idx]
        abs_x = px_x + left
        abs_y = px_y + top
        yv = np.asarray(y_axis.to_value(abs_y), float)
        if x_axis is not None:
            xv = np.asarray(x_axis.to_value(abs_x), float)
            x_rel = False
        else:
            xv = px_x / max(W - 1, 1)
            x_rel = True
        clipped = bool(np.any(px_y <= 1) or (kind == "line" and np.any(px_y >= H - 2)))
        coverage = cols.size / max(cols[-1] - cols[0] + 1, 1) if kind != "bar" else 1.0
        conf = y_axis.confidence * (0.6 + 0.4 * coverage) * (0.7 if clipped else 1.0)
        if x_axis is not None:
            conf *= 0.5 + 0.5 * x_axis.confidence
        else:
            conf *= 0.85
        mid_h = ((lo + hi) / 2) % 360 if lo < hi else ((lo + hi + 360) / 2) % 360
        sel = cm & _in_hue(h, lo, hi)
        color = crop[sel].mean(axis=0) if sel.any() else np.array([0.5, 0.5, 0.5])
        series.append(
            SeriesEstimate(
                name=f"series@{int(mid_h)}°",
                color="#{:02x}{:02x}{:02x}".format(*(np.clip(color, 0, 1) * 255).astype(int)),
                kind=kind,
                x=[float(a) for a in xv],
                y=[float(b) for b in yv],
                x_is_relative=x_rel,
                latest=float(yv[np.argmax(xv)]) if yv.size else None,
                minimum=float(np.min(yv)) if yv.size else None,
                maximum=float(np.max(yv)) if yv.size else None,
                clipped=clipped,
                confidence=float(np.clip(conf, 0, 1)),
            )
        )
    return series


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return [r for r in runs if r[1] - r[0] >= 1]


DASH_CHARS = set("-—–_~=")


def is_dash_token(text: str) -> bool:
    t = text.strip()
    return bool(t) and all(c in DASH_CHARS for c in t)


def _hex_to_hue(hex_color: str) -> float:
    rgb = np.array([[int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]], np.float32)
    h, _, _ = _rgb_to_hsv(rgb)
    return float(h[0])


def _assign_legend(rgb: np.ndarray, chart: ChartEstimate, lines: list[list[OcrWord]], line_reader: Any = None) -> None:
    left, top, right, bottom = chart.bbox
    hues = {id(s): _hex_to_hue(s.color) for s in chart.series}
    for line in lines:
        first = line[0]
        if not (left - 40 <= first.left <= right + 260 and top - 90 <= first.cy <= bottom + 60):
            continue
        swatch = first if is_dash_token(first.text) else None
        words = [w for w in line if not is_dash_token(w.text)]
        if swatch is not None and not words and line_reader is not None:
            box = (swatch.right + 2, int(swatch.cy - 1.6 * max(swatch.height, 8)), min(swatch.right + 260, rgb.shape[1]), int(swatch.cy + 1.6 * max(swatch.height, 8)))
            txt = line_reader(box)
            if txt:
                words = [OcrWord(txt, 0.6, box[0], box[1], box[2] - box[0], box[3] - box[1])]
        text = " ".join(w.text for w in words if not w.text.replace(".", "").isdigit()).strip()
        if not text or parse_number(text) is not None or not words:
            continue
        if swatch is not None:  # OCR read the legend line sample as a dash "word": sample its colour
            x0, x1 = swatch.left, swatch.right
        else:
            x0, x1 = max(words[0].left - 30, 0), max(words[0].left - 2, 0)
        ref = words[0]
        y0, y1 = max(int(ref.cy - ref.height / 2), 0), int(ref.cy + ref.height / 2) + 1
        patch = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
        if patch.size == 0:
            continue
        h, s, v = _rgb_to_hsv(patch)
        m = (s > 0.3) & (v > 0.25)
        if m.sum() < 4:
            continue
        hue = float(np.median(h[m]))
        best, best_d = None, 20.0
        for ser in chart.series:
            d = abs((hues[id(ser)] - hue + 180) % 360 - 180)
            if d < best_d:
                best, best_d = ser, d
        if best is not None and best.name.startswith("series@"):
            best.name = text[:60]


# ------------------------------------------------------------------ KPIs
def _extract_kpis(lines: list[list[OcrWord]], used: set[int]) -> list[KpiEstimate]:
    kpis: list[KpiEstimate] = []
    for li, line in enumerate(lines):
        for wi, w in enumerate(line):
            if id(w) in used:
                continue
            val = parse_number(w.text)
            if val is None:
                continue
            unit = ""
            m = re.search(r"(%|[A-Za-z°Ω]{1,4})$", w.text)
            if m and m.group(1).lower() in UNIT_TOKENS:
                unit = m.group(1)
            elif wi + 1 < len(line) and line[wi + 1].text.lower().strip(".,:") in UNIT_TOKENS:
                unit = line[wi + 1].text.strip(".,:")
            # Label on the same line: contiguous non-numeric words immediately before the number.
            label_parts: list[str] = []
            right_edge = w.left
            for x in reversed(line[:wi]):
                if parse_number(x.text) is not None or x.text.lower() in UNIT_TOKENS or right_edge - x.right > 3 * x.height:
                    break
                label_parts.insert(0, x.text)
                right_edge = x.left
            label = " ".join(label_parts).strip(" :=-")
            if not label:  # tile layout: label on the line above, horizontally overlapping
                for prev in reversed(lines[:li]):
                    if not prev or w.top - max(p.bottom for p in prev) > 2.5 * w.height:
                        continue
                    over = [p for p in prev if p.left < w.right + 60 and p.right > w.left - 40 and parse_number(p.text) is None]
                    if over:
                        label = " ".join(p.text for p in over).strip(" :=-")
                        break
            if not label or len(label) > 60:
                continue
            kpis.append(
                KpiEstimate(
                    label=label,
                    value=val,
                    unit=unit,
                    confidence=round(w.conf * 0.9, 3),
                    bbox=(w.left, w.top, w.right, w.bottom),
                    raw_text=" ".join(x.text for x in line),
                )
            )
    return kpis


# ------------------------------------------------------------------ public API
def analyze_chart_image(
    img: Image.Image,
    words: list[OcrWord],
    captured_at: datetime | None = None,
    line_reader: Any = None,
) -> tuple[list[ChartEstimate], list[KpiEstimate], list[str]]:
    """``line_reader(box) -> str`` optionally re-reads a single text line (used for
    legend labels the full-page OCR pass missed)."""
    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    warnings: list[str] = []
    nums, times = _numeric_ticks(words)
    used: set[int] = set()
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    candidates: list[tuple[list[_Tick], str]] = [(g, "time") for g in _x_axis_groups(times)]
    candidates += [(g, "linear") for g in _x_axis_groups(nums)]
    x_groups: list[tuple[list[_Tick], str]] = []
    for g, scale in candidates:
        # A row of numbers is only an axis if its values are monotonic in x and calibrate well
        # (this keeps rows of KPI tiles from being mistaken for tick labels).
        cal = fit_axis("x", [t.word.cx for t in g], [t.value for t in g], [t.word.conf for t in g], "time" if scale == "time" else None)
        if cal is None or cal.inliers < max(3, int(0.6 * len(g))) or cal.reversed:
            continue
        x_groups.append((g, scale))
        used.update(id(t.word) for t in g)
    y_groups = _y_axis_groups(nums, used)
    charts: list[ChartEstimate] = []
    lines = group_lines(words)
    for gi, yg in enumerate(y_groups):
        y_cal = fit_axis("y", [_snap_y(gray, t.word) for t in yg], [t.value for t in yg], [t.word.conf for t in yg])
        if y_cal is None or y_cal.confidence < 0.2:
            continue
        y_left = min(t.word.left for t in yg)
        y_right = max(t.word.right for t in yg)
        y_top = min(t.word.cy for t in yg)
        y_bot = max(t.word.cy for t in yg)
        spacing = (y_bot - y_top) / max(len(yg) - 1, 1)
        # Pair with the closest x-axis row below the lowest y tick and to the right of the y labels.
        best_x, best_d = None, float("inf")
        for xg, scale in x_groups:
            xcy = float(np.median([t.word.cy for t in xg]))
            xl = min(t.word.left for t in xg)
            if xcy < y_bot - 4 or xcy - y_bot > 3 * spacing + 30 or xl < y_left - 30:
                continue
            if xl > y_right + 0.6 * W:
                continue
            d = xcy - y_bot + 0.1 * abs(xl - y_right)
            if d < best_d:
                best_x, best_d = (xg, scale), d
        x_cal = None
        if best_x is not None:
            xg, scale = best_x
            x_cal = fit_axis("x", [_snap_x(gray, t.word) for t in xg], [t.value for t in xg], [t.word.conf for t in xg], "time" if scale == "time" else None)
            used.update(id(t.word) for t in xg)
            x_right = max(t.word.cx for t in xg)
            x_spacing = (x_right - min(t.word.cx for t in xg)) / max(len(xg) - 1, 1)
            plot_right = int(min(W - 1, x_right + 0.5 * x_spacing))
            plot_bottom = int(min(t.word.top for t in xg) - 2)
        else:
            plot_right = W - 1
            # Stop at the next chart's y-axis labels on the same band, if any.
            for other in y_groups:
                ol = min(t.word.left for t in other)
                if other is not yg and ol > y_right and min(t.word.cy for t in other) < y_bot and max(t.word.cy for t in other) > y_top:
                    plot_right = min(plot_right, ol - 10)
            plot_bottom = int(min(H - 1, y_bot + 0.5 * spacing))
            warnings.append(f"Chart {gi + 1}: no numeric x-axis found; x positions are relative (0–1).")
        plot_top = _find_plot_top(gray, int(y_right + 3), plot_right, y_top, spacing, lines)
        plot = (int(y_right + 3), plot_top, plot_right, plot_bottom)
        if plot[2] - plot[0] < 30 or plot[3] - plot[1] < 30:
            continue
        used.update(id(t.word) for t in yg)
        inside_boxes = [
            (w.left, w.top, w.right, w.bottom)
            for w in words
            if plot[0] <= w.left <= plot[2] and plot[1] <= w.top <= plot[3]
            and len(w.text) >= 2 and w.conf >= 0.5 and not is_dash_token(w.text)
        ]
        # Legend swatches (read by OCR as dash "words") are masked exactly, without the text margin.
        swatches = [
            (w.left - 2, w.top - 2, w.right + 2, w.bottom + 2)
            for w in words
            if plot[0] <= w.left <= plot[2] and plot[1] <= w.top <= plot[3] and is_dash_token(w.text)
        ]
        series = _extract_series(rgb, plot, inside_boxes, y_cal, x_cal, swatches)
        title = _find_title(lines, plot, used, y_top)
        unit = _find_unit(title, yg)
        chart = ChartEstimate(
            id=f"chart{gi + 1}",
            bbox=plot,
            title=title,
            chart_type=_majority([s.kind for s in series]) if series else "unknown",
            x_axis=x_cal,
            y_axis=y_cal,
            series=series,
            unit=unit,
            confidence=float(np.mean([s.confidence for s in series])) if series else y_cal.confidence * 0.3,
        )
        if not series:
            chart.warnings.append("No coloured series found (grey/black series are not separated from gridlines).")
        if y_cal.reversed:
            chart.warnings.append("Y-axis is reversed (values increase downward).")
        if y_cal.scale == "log":
            chart.warnings.append("Y-axis detected as logarithmic.")
        if y_cal.inliers < len(yg):
            chart.warnings.append(f"{len(yg) - y_cal.inliers} y-tick label(s) were inconsistent and ignored (likely OCR misreads).")
        if any(s.clipped for s in series):
            chart.warnings.append("A series touches the plot edge; values beyond the visible range are not known.")
        _assign_legend(rgb, chart, lines, line_reader)
        if len(chart.series) == 1 and chart.series[0].name.startswith("series@") and title:
            chart.series[0].name = title
        # Time x-axis -> absolute timestamps (seconds since midnight on the capture date)
        if x_cal is not None and x_cal.scale == "time" and captured_at is not None:
            chart.warnings.append("X-axis read as clock time on the capture date (local time of the screen is assumed UTC unless configured).")
        charts.append(chart)
    for c in charts:
        for w in words:
            if c.bbox[0] <= w.cx <= c.bbox[2] and c.bbox[1] <= w.cy <= c.bbox[3]:
                used.add(id(w))
    kpis = _extract_kpis(lines, used)
    return charts, kpis, warnings


def _snap_y(gray: np.ndarray, w: OcrWord) -> float:
    """Snap a y tick label to the tick mark / gridline next to it (OCR boxes are
    a pixel or two off the real tick position)."""
    H, W = gray.shape
    y0, y1 = max(int(w.cy - 8), 0), min(int(w.cy + 9), H)
    x0, x1 = min(w.right + 1, W - 1), min(w.right + 16, W)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return w.cy
    dark = (patch < 200).sum(axis=1)
    if dark.max() >= 3 and dark.max() >= 2 * np.median(dark) + 1:
        rows = np.where(dark == dark.max())[0]
        return float(y0 + rows.mean())
    return w.cy


def _snap_x(gray: np.ndarray, w: OcrWord) -> float:
    H, W = gray.shape
    x0, x1 = max(int(w.cx - 8), 0), min(int(w.cx + 9), W)
    y0, y1 = max(w.top - 14, 0), max(w.top - 1, 0)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return w.cx
    dark = (patch < 200).sum(axis=0)
    if dark.max() >= 3 and dark.max() >= 2 * np.median(dark) + 1:
        cols = np.where(dark == dark.max())[0]
        return float(x0 + cols.mean())
    return w.cx


def _find_plot_top(gray: np.ndarray, left: int, right: int, y_top: float, spacing: float, lines: list[list[OcrWord]]) -> int:
    """Upper edge of the plot area: a frame/border line if one exists within one tick
    spacing above the top tick, otherwise bounded by the title text above."""
    lo = int(max(0, y_top - 1.2 * spacing))
    hi = int(y_top)
    seg = gray[lo : hi + 1, left : max(right, left + 1)]
    if seg.size:
        frac = (seg < 225).mean(axis=1)
        rows = np.where(frac > 0.7)[0]
        if rows.size:
            return int(lo + rows.min() + 1)
    bound = 0
    for line in lines:
        bottom = max(w.bottom for w in line)
        if bottom < y_top - 6 and line[-1].right >= left and line[0].left <= right:
            bound = max(bound, bottom + 3)
    return int(max(bound, y_top - 0.6 * spacing, 0))


def _find_title(lines: list[list[OcrWord]], plot: tuple[int, int, int, int], used: set[int], y_top: float) -> str:
    """The chart title is the nearest mostly-textual line above the plot area that
    overlaps it horizontally (KPI tiles with large numbers above a chart are not titles)."""
    left, _, right, _ = plot
    best, best_d = "", float("inf")
    for line in lines:
        cy = float(np.mean([w.cy for w in line]))
        if not (y_top - 120 <= cy <= y_top - 6):
            continue
        if line[-1].right < left - 60 or line[0].left > right:
            continue
        words = [w for w in line if id(w) not in used]
        if not words:
            continue
        numeric = sum(parse_number(w.text) is not None for w in words)
        text = " ".join(w.text for w in words).strip()
        if len(text) < 3 or numeric / len(words) >= 0.34:
            continue
        d = y_top - cy
        if d < best_d:
            best, best_d = text, d
    return best[:80]


def _find_unit(title: str, yg: list[_Tick]) -> str:
    m = re.search(r"[\(\[]([^\)\]]{1,8})[\)\]]", title)
    if m:
        return m.group(1).strip()
    for t in yg:
        m2 = re.search(r"(%|[A-Za-z°Ω]{1,4})$", t.word.text)
        if m2 and m2.group(1).lower() in UNIT_TOKENS:
            return m2.group(1)
    return ""


def _majority(items: list[str]) -> str:
    return max(set(items), key=items.count)


def time_axis_to_datetimes(seconds_of_day: list[float], captured_at: datetime) -> list[datetime]:
    """Map clock-time x values to datetimes on the capture date; times later than
    the capture are assumed to belong to the previous day (charts don't show the future)."""
    day = captured_at.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for s in seconds_of_day:
        dt = day + timedelta(seconds=float(s))
        if dt > captured_at + timedelta(minutes=5):
            dt -= timedelta(days=1)
        out.append(dt)
    return out


def chart_to_dict(chart: ChartEstimate) -> dict[str, Any]:
    d = asdict(chart)
    return d
