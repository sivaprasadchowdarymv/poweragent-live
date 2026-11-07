"""Chart-model adapter (IBM Granite Vision 4.1 or OpenGVLab ChartAst).

Chart-specialised vision-language models run as a separate local inference
server (``chart_model_server/``) with one shared API. This module:

* talks to that server (health, chart-to-table, chart QA),
* crops each chart the pixel reader found (the models are trained on single charts),
* parses the table output — CSV (Granite ``<chart2csv>``) or Markdown (ChartAst) —
  into numeric series,
* cross-checks those numbers against the pixel-calibration reading of the same
  chart and derives an agreement-based confidence.

Everything a chart model returns is a *model reading* of an image: it is stored
as an estimate under its own source (``screen_chart_model``) and is never merged
with the pixel reader or with file data.
"""

from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests
from PIL import Image

from vision.chart_reader import ChartEstimate
from vision.ocr import parse_number, parse_time_of_day

GENERIC_SERIES = {"value", "values", "characteristic", "data", "y", "series", "series 1", "amount", "count", "number", "total"}


class ChartModelError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # offline | busy | bad_request | server | timeout | bad_response


# ---------------------------------------------------------------------------- client
class ChartModelClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8600", timeout: float = 180.0, connect_timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.session = requests.Session()

    def health(self) -> dict[str, Any]:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=(self.connect_timeout, 10))
            data = r.json() if r.ok else {"ok": False, "error": f"HTTP {r.status_code}"}
            return data
        except requests.RequestException as exc:
            return {"ok": False, "error": f"Cannot reach the chart-model server at {self.base_url}: {exc.__class__.__name__}"}
        except ValueError:
            return {"ok": False, "error": "The chart-model server returned an invalid response."}

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.session.post(f"{self.base_url}{path}", json=body, timeout=(self.connect_timeout, self.timeout))
        except requests.ConnectionError as exc:
            raise ChartModelError("offline", f"Cannot reach the chart-model server at {self.base_url}.") from exc
        except requests.Timeout as exc:
            raise ChartModelError("timeout", f"The chart model did not answer within {self.timeout:.0f}s.") from exc
        if r.status_code == 503:
            raise ChartModelError("busy", "The chart model is busy with another chart.")
        if r.status_code == 400:
            raise ChartModelError("bad_request", r.text[:300])
        if not r.ok:
            raise ChartModelError("server", f"chart-model server error HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError as exc:
            raise ChartModelError("bad_response", "The chart model returned invalid JSON.") from exc

    def chart_to_table(self, image: Image.Image) -> dict[str, Any]:
        return self._post("/v1/chart_to_table", {"image": encode_image(image)})

    def chart_qa(self, image: Image.Image, question: str) -> dict[str, Any]:
        return self._post("/v1/chart_qa", {"image": encode_image(image), "question": question[:500]})


def encode_image(img: Image.Image, max_side: int = 1344) -> str:
    """PNG keeps thin chart lines and small digits sharp (JPEG blurs them)."""
    im = img.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def crop_for_chart(img: Image.Image, chart: ChartEstimate | None) -> Image.Image:
    """Chart models are trained on single charts: crop the plot area plus room for the
    title, axis labels and a right-hand legend."""
    if chart is None:
        return img
    left, top, right, bottom = chart.bbox
    box = (max(0, left - 110), max(0, top - 120), min(img.width, right + 300), min(img.height, bottom + 70))
    return img.crop(box)


# ---------------------------------------------------------------------------- table parsing
@dataclass
class ChartTable:
    title: str
    header: list[str]
    rows: list[list[str]]
    series: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    x_kind: str = "category"  # category | number | time
    raw: str = ""
    parse_notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.series) and any(len(v) for v in self.series.values())

    def summary(self, max_rows: int = 25) -> dict[str, Any]:
        return {
            "title": self.title,
            "x_kind": self.x_kind,
            "series": {k: [{"x": x, "value": v} for x, v in pts[-max_rows:]] for k, pts in self.series.items()},
            "notes": self.parse_notes,
        }


def _cell_number(cell: str) -> float | None:
    c = cell.strip().replace("−", "-").replace("$", "").replace("€", "").replace("£", "").replace("₹", "")
    c = re.sub(r"\s+", "", c)
    if not c or c in ("-", "--", "n/a", "N/A", "nan"):
        return None
    return parse_number(c)


def _strip_fence(text: str) -> str:
    """Remove ```csv / ```markdown fences that models often add."""
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def _looks_like_csv(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2 or sum("|" in ln for ln in lines) >= len(lines) / 2:
        return False
    return sum("," in ln or "\t" in ln or ";" in ln for ln in lines) >= 0.8 * len(lines)


def _csv_to_pipes(text: str) -> str:
    import csv as _csv

    sample = "\n".join(text.strip().splitlines()[:5])
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t")
        delim = dialect.delimiter
    except _csv.Error:
        delim = ","
    rows = list(_csv.reader(io.StringIO(text.strip()), delimiter=delim))
    return "\n".join("| " + " | ".join(c.replace("|", "/").strip() for c in r) + " |" for r in rows if any(c.strip() for c in r))


def parse_chart_table(text: str) -> ChartTable:
    """Parse a chart model's table output (CSV, Markdown or linearised)."""
    return parse_markdown_table(text)


def parse_markdown_table(text: str) -> ChartTable:
    """Parse The chart model's chart-to-table output.

    Accepts a Markdown pipe table (with or without outer pipes and separator
    row), the linearised DePlot/UniChart style ``Year | Sales <0x0A> 2019 | 12``,
    and an optional ``TITLE | ...`` first line. Orientation is detected: if the
    header row is numeric (e.g. years across) the table is transposed so each
    series is a row.
    """
    raw = text or ""
    t = _strip_fence(raw).replace("<0x0A>", "\n").replace("\\n", "\n").replace("&&", "\n")
    if _looks_like_csv(t):
        t = _csv_to_pipes(t)
    lines = [ln.strip() for ln in t.splitlines() if "|" in ln]
    rows: list[list[str]] = []
    title = ""
    for ln in lines:
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", ln):
            continue  # separator row
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0].upper().startswith("TITLE"):
            title = " ".join(c for c in cells[1:] if c).strip() or title
            continue
        if any(cells):
            rows.append(cells)
    table = ChartTable(title=title, header=[], rows=[], raw=raw[:4000])
    if len(rows) < 2:
        table.parse_notes.append("No table with a header and at least one data row was found.")
        return table
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    header_numeric = sum(_cell_number(h) is not None for h in header[1:]) >= max(1, (len(header) - 1) // 2 + 1) and len(header) > 2
    first_col_numeric = sum(_cell_number(r[0]) is not None for r in body) == len(body)
    if header_numeric and not first_col_numeric:
        # Transposed layout: series are rows, x values across the header.
        table.parse_notes.append("Table was transposed (series in rows).")
        x_labels = header[1:]
        for r in body:
            pts = [(x, v) for x, c in zip(x_labels, r[1:], strict=False) if (v := _cell_number(c)) is not None]
            if pts:
                table.series[r[0] or f"series {len(table.series) + 1}"] = pts
    else:
        x_labels = [r[0] for r in body]
        for j, name in enumerate(header[1:], start=1):
            pts = [(x, v) for x, r in zip(x_labels, body, strict=True) if (v := _cell_number(r[j])) is not None]
            if pts:
                table.series[name or f"series {j}"] = pts
        if not table.series and width == 1:
            table.parse_notes.append("Single-column table has no values.")
    xs = [x for pts in table.series.values() for x, _ in pts]
    if xs and all(parse_time_of_day(x) is not None for x in xs):
        table.x_kind = "time"
    elif xs and all(_cell_number(x) is not None for x in xs):
        table.x_kind = "number"
    table.header, table.rows = header, body
    skipped = sum(len(body) for _ in table.series) - sum(len(p) for p in table.series.values())
    if skipped > 0:
        table.parse_notes.append(f"{skipped} non-numeric cell(s) were ignored.")
    return table


def x_value(label: str, kind: str) -> float | None:
    if kind == "time":
        return parse_time_of_day(label)
    if kind == "number":
        return _cell_number(label)
    return None


# ---------------------------------------------------------------------------- cross-check
@dataclass
class CrossCheck:
    status: str  # agree | partial | disagree | not_comparable
    compared_points: int = 0
    mean_abs_error: float | None = None
    max_abs_error: float | None = None
    error_pct_of_axis: float | None = None
    agreement_pct: float | None = None
    pairs: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "compared_points": self.compared_points,
            "mean_abs_error": self.mean_abs_error,
            "max_abs_error": self.max_abs_error,
            "error_pct_of_axis": self.error_pct_of_axis,
            "agreement_pct": self.agreement_pct,
            "series_pairs": self.pairs[:8],
            "note": self.note,
        }


def _name_score(a: str, b: str) -> float:
    import difflib

    a, b = a.lower().replace(" ", ""), b.lower().replace(" ", "")
    return difflib.SequenceMatcher(None, a, b).ratio()


def cross_check(table: ChartTable, chart: ChartEstimate | None, tol_pct: float = 3.0) -> CrossCheck:
    """Compare The chart model's numbers with the pixel reader's numbers for the same chart.
    Tolerance is a percentage of the y-axis span, so it is scale-independent."""
    if chart is None or not chart.series or not table.ok:
        return CrossCheck("not_comparable", note="No pixel reading of this chart to compare with.")
    ticks = [v for _, v in chart.y_axis.ticks]
    span = (max(ticks) - min(ticks)) if len(ticks) >= 2 else None
    if not span:
        return CrossCheck("not_comparable", note="The pixel reader has no usable y-axis range.")
    diffs: list[float] = []
    pairs = []
    used: set[str] = set()
    for ps in chart.series:
        cands = [k for k in table.series if k not in used]
        if not cands:
            break
        best = max(cands, key=lambda k: _name_score(k, ps.name))
        if len(chart.series) > 1 and _name_score(best, ps.name) < 0.5:
            # names don't help: pick the table series whose values are closest
            best = min(cands, key=lambda k: abs(np.median([v for _, v in table.series[k]]) - np.median(ps.y)))
        used.add(best)
        pts = table.series[best]
        d: list[float] = []
        if chart.chart_type == "bar" and len(ps.y) == len(pts):
            d = [abs(a - v) for a, (_, v) in zip(ps.y, pts, strict=True)]
        elif chart.x_axis is not None and not ps.x_is_relative and table.x_kind in ("number", "time"):
            xs = np.array(ps.x)
            order = np.argsort(xs)
            for lab, v in pts:
                xv = x_value(lab, table.x_kind)
                if xv is None or xv < xs.min() or xv > xs.max():
                    continue
                d.append(abs(float(np.interp(xv, xs[order], np.array(ps.y)[order])) - v))
        else:
            # Unknown x mapping: compare first/last points and extremes only.
            tv = [v for _, v in pts]
            d = [abs(tv[-1] - (ps.latest or ps.y[-1])), abs(tv[0] - ps.y[int(np.argmin(ps.x))]), abs(max(tv) - (ps.maximum or max(ps.y))), abs(min(tv) - (ps.minimum or min(ps.y)))]
        if d:
            diffs.extend(d)
            pairs.append({"pixel_series": ps.name, "chart_model_series": best, "points": len(d), "mean_abs_error": float(np.mean(d))})
    if not diffs:
        return CrossCheck("not_comparable", note="The chart model's x values could not be matched to the chart's x-axis.")
    arr = np.array(diffs)
    within = float(np.mean(arr <= tol_pct / 100 * span) * 100)
    err_pct = float(arr.mean() / span * 100)
    status = "agree" if within >= 80 else "partial" if within >= 40 else "disagree"
    return CrossCheck(status, len(arr), float(arr.mean()), float(arr.max()), round(err_pct, 2), round(within, 1), pairs)


def fused_confidence(check: CrossCheck, base: float = 0.65) -> float:
    return {"agree": 0.9, "partial": 0.6, "disagree": 0.35}.get(check.status, base)


# ---------------------------------------------------------------------------- per-frame run
@dataclass
class ChartModelReading:
    model: str
    chart_id: str
    chart_title: str
    table: ChartTable
    check: CrossCheck
    ms: float
    error: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "chart_title": self.chart_title,
            "table": self.table.summary() if self.table.ok else None,
            "cross_check_with_pixel_reader": self.check.to_dict(),
            "model": self.model,
            "inference_ms": round(self.ms),
            "error": self.error,
            "status": "estimated (chart-model reading)",
        }


def read_frame(client: ChartModelClient, image: Image.Image, charts: list[ChartEstimate], max_charts: int = 3) -> list[ChartModelReading]:
    """Run chart-to-table on each detected chart (or the whole image when the
    pixel reader found none)."""
    targets: list[ChartEstimate | None] = list(charts[:max_charts]) or [None]
    out = []
    model = "chart model"
    for ch in targets:
        crop = crop_for_chart(image, ch)
        t0 = time.perf_counter()
        try:
            res = client.chart_to_table(crop)
            model = str(res.get("model") or model)
            table = parse_chart_table(res.get("text", ""))
            err = "" if table.ok else "The chart model's output contained no numeric table."
        except ChartModelError as exc:
            table, err = ChartTable("", [], []), str(exc)
            if exc.kind in ("offline", "busy"):
                out.append(ChartModelReading(model, ch.id if ch else "frame", ch.title if ch else "", table, CrossCheck("not_comparable"), (time.perf_counter() - t0) * 1000, err))
                break
        check = cross_check(table, ch) if table.ok else CrossCheck("not_comparable")
        if not table.title and ch is not None:
            table.title = ch.title
        out.append(ChartModelReading(model, ch.id if ch else "frame", ch.title if ch else table.title, table, check, (time.perf_counter() - t0) * 1000, err))
    return out


def series_metric_name(series_name: str, chart_title: str) -> str:
    name = series_name.strip()
    if not name or name.lower() in GENERIC_SERIES:
        name = chart_title.strip() or "chart value"
    return re.sub(r"\s*[\(\[][^\)\]]{1,12}[\)\]]\s*$", "", name).strip() or name
