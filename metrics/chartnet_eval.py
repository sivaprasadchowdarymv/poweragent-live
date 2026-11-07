"""ChartNet benchmark: measure how accurately each chart reader extracts data.

ChartNet (IBM, https://huggingface.co/datasets/ibm-granite/ChartNet) pairs
every chart image with the exact data table (``csv``) used to draw it, across
24 chart types and several plotting libraries. This script samples charts,
runs the app's readers on them and scores the extracted numbers against that
ground truth:

* ``pixel`` — the built-in OCR + axis-calibration reader (no model needed)
* ``model`` — a chart-model server (Granite Vision 4.1 or ChartAst) at --model-url

Scores (per chart, then averaged overall / per chart type / per library):

* cell recall@5%   share of ground-truth values the reader got within 5 %
* cell precision@5% share of the reader's values that match a ground-truth value within 5 %
* F1@5% and F1@10%
* extraction rate  share of charts where the reader returned any numbers

The per-chart-type F1 is saved as ``priors.json``; the app uses it to scale the
confidence of each reader's live readings by what was actually measured.

Free data only by default: ``core_permissive`` is CDLA-Permissive-2.0. The
``human_verified`` / ``core`` subsets carry IBM's notice that they "may not be
used for commercial exploitation or commercial deployment" — fine for private
evaluation, flagged in the report.

Usage (from the project folder):
    python -m metrics.chartnet_eval --samples 200
    python -m metrics.chartnet_eval --samples 300 --readers pixel,model --model-url http://127.0.0.1:8600
    python -m metrics.chartnet_eval --parquet D:\\data\\core_permissive_000.parquet --samples 500
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.chart_model import ChartModelClient, ChartModelError, ChartTable, parse_chart_table  # noqa: E402
from vision.chart_reader import ChartEstimate, analyze_chart_image  # noqa: E402
from vision.ocr import OCRUnavailable, parse_time_of_day, read_line, run_ocr  # noqa: E402

DATASET = "ibm-granite/ChartNet"
PERMISSIVE = {"core_permissive"}
NONCOMMERCIAL_NOTE = "This ChartNet subset carries IBM's notice: it may not be used for commercial exploitation or commercial deployment."


# =========================================================================== data loading
@dataclass
class Sample:
    id: str
    image: Image.Image
    truth: ChartTable
    chart_type: str
    library: str


def _row_image(row: dict[str, Any]) -> Image.Image | None:
    for key in ("image", "img", "png"):
        v = row.get(key)
        if v is None:
            continue
        if isinstance(v, Image.Image):
            return v.convert("RGB")
        if isinstance(v, dict) and v.get("bytes"):
            return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(v)).convert("RGB")
    return None


def _row_to_sample(row: dict[str, Any]) -> Sample | None:
    csv_text = row.get("csv") or row.get("table") or row.get("data") or ""
    img = _row_image(row)
    if img is None or not isinstance(csv_text, str) or not csv_text.strip():
        return None
    truth = parse_chart_table(csv_text)
    if not truth.ok:
        return None
    return Sample(str(row.get("id", "")), img, truth, str(row.get("chart_type", "unknown")), str(row.get("library", "unknown")))


def load_samples(n: int, subset: str = "core_permissive", split: str = "train", parquet: list[str] | None = None, seed: int = 0, chart_types: set[str] | None = None, max_pixels: int = 6_000_000) -> list[Sample]:
    """Sample ``n`` usable charts. Uses local parquet files if given, otherwise
    streams from Hugging Face (only the rows read are downloaded)."""
    rows_iter: Any
    if parquet:
        import pyarrow.parquet as pq

        def gen() -> Any:
            for path in parquet:
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=64):
                    yield from batch.to_pylist()

        rows_iter = gen()
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install the free 'datasets' package first:  pip install datasets   (or pass --parquet with a downloaded file)") from exc
        ds = load_dataset(DATASET, subset, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=max(1000, 5 * n))
        rows_iter = iter(ds)
    out: list[Sample] = []
    per_type: dict[str, int] = defaultdict(int)
    cap = max(3, math.ceil(n / 8))  # keep the sample diverse across chart types
    rng = random.Random(seed)
    for row in rows_iter:
        ct = str(row.get("chart_type", "unknown"))
        if chart_types and ct not in chart_types:
            continue
        if per_type[ct] >= cap and rng.random() < 0.9:
            continue
        s = _row_to_sample(row)
        if s is None or s.image.width * s.image.height > max_pixels:
            continue
        out.append(s)
        per_type[ct] += 1
        if len(out) >= n:
            break
    return out


# =========================================================================== readers
@dataclass
class Prediction:
    series: dict[str, list[tuple[str | None, float]]] = field(default_factory=dict)  # x label may be None (positional)
    positional: bool = False  # x positions are fractions 0..1 / indices rather than labels
    fractions: dict[str, list[float]] = field(default_factory=dict)
    error: str = ""
    ms: float = 0.0


def _main_chart(charts: list[ChartEstimate]) -> ChartEstimate | None:
    with_series = [c for c in charts if c.series]
    if not with_series:
        return None
    return max(with_series, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))


def read_pixel(img: Image.Image) -> Prediction:
    t0 = time.perf_counter()
    try:
        words = run_ocr(img)
    except OCRUnavailable as exc:
        return Prediction(error=str(exc))
    charts, _, _ = analyze_chart_image(img, words, line_reader=lambda b: read_line(img, b))
    chart = _main_chart(charts)
    p = Prediction(ms=(time.perf_counter() - t0) * 1000)
    if chart is None:
        p.error = "no chart/series detected"
        return p
    p.positional = True
    for s in chart.series:
        xs = np.array(s.x, float)
        order = np.argsort(xs)
        xs, ys = xs[order], np.array(s.y, float)[order]
        if chart.x_axis is not None and not s.x_is_relative:
            p.series[s.name] = [(f"{x:g}", float(y)) for x, y in zip(xs, ys, strict=True)]
            span = xs.max() - xs.min() or 1.0
            p.fractions[s.name] = list((xs - xs.min()) / span)
        else:
            p.series[s.name] = [(None, float(y)) for y in ys]
            p.fractions[s.name] = list(xs)
        if chart.chart_type == "bar":
            p.fractions[s.name] = list(np.linspace(0, 1, len(ys))) if len(ys) > 1 else [0.0]
    return p


def read_model(img: Image.Image, client: ChartModelClient) -> Prediction:
    t0 = time.perf_counter()
    try:
        res = client.chart_to_table(img)
    except ChartModelError as exc:
        return Prediction(error=str(exc), ms=(time.perf_counter() - t0) * 1000)
    t = parse_chart_table(res.get("text", ""))
    p = Prediction(ms=(time.perf_counter() - t0) * 1000)
    if not t.ok:
        p.error = "no numeric table in model output"
        return p
    p.series = {k: [(x, v) for x, v in pts] for k, pts in t.series.items()}
    return p


# =========================================================================== scoring
def _norm(label: str | None) -> str:
    return re.sub(r"[^a-z0-9.]+", "", (label or "").lower())


def _as_number(label: str | None) -> float | None:
    if label is None:
        return None
    t = parse_time_of_day(label)
    if t is not None:
        return t
    try:
        return float(label.replace(",", "").strip())
    except ValueError:
        return None


def _within(pred: float, truth: float, tol: float, floor: float) -> bool:
    return abs(pred - truth) <= max(tol * abs(truth), floor)


def _values_for(truth_pts: list[tuple[str, float]], pred: Prediction, name: str) -> list[float | None]:
    """Predicted value for each ground-truth row of one series."""
    pts = pred.series[name]
    labelled = [(x, v) for x, v in pts if x is not None]
    out: list[float | None] = []
    if labelled and not pred.positional:
        by_label = {_norm(x): v for x, v in labelled}
        nums = [(n, v) for x, v in labelled if (n := _as_number(x)) is not None]
        for x, _ in truth_pts:
            v = by_label.get(_norm(x))
            if v is None and nums and (tx := _as_number(x)) is not None:
                xs = np.array([a for a, _ in nums])
                if xs.min() <= tx <= xs.max():
                    v = float(np.interp(tx, xs, [b for _, b in nums]))
            out.append(v)
        if all(v is None for v in out) and len(labelled) == len(truth_pts):
            out = [v for _, v in labelled]  # labels differ in wording but rows line up
        return out
    # Positional readers: numeric truth x -> interpolate on numeric axis; else by order.
    txs = [_as_number(x) for x, _ in truth_pts]
    if labelled and all(t is not None for t in txs):
        xs = np.array([_as_number(x) for x, _ in labelled], float)
        ys = np.array([v for _, v in labelled], float)
        return [float(np.interp(t, xs, ys)) if t is not None and xs.min() - 1e-9 <= t <= xs.max() + 1e-9 else None for t in txs]
    fr = np.array(pred.fractions.get(name, []), float)
    ys = np.array([v for _, v in pts], float)
    if fr.size == 0:
        return [None] * len(truth_pts)
    if fr.size == len(truth_pts):
        return list(ys)
    target = np.linspace(fr.min(), fr.max(), len(truth_pts)) if len(truth_pts) > 1 else np.array([fr.max()])
    return [float(np.interp(t, fr, ys)) for t in target]


def score(truth: ChartTable, pred: Prediction, tol: float = 0.05) -> dict[str, Any]:
    """Cell-level precision/recall against the ground-truth table.

    * Recall: share of ground-truth values the reader reproduced within ``tol``.
    * Precision: share of the values the reader offered *at the ground-truth
      positions* that are correct, plus a penalty for every extra series it
      invented. Dense line readers are therefore not punished for returning
      hundreds of samples, only for wrong ones.
    * Near-zero values use 1 % of that series' own largest value as the
      tolerance floor (per series, so mixed-scale charts are not too lenient).
    """
    gt_series = list(truth.series)
    floors = {g: 0.01 * (max(abs(v) for _, v in pts) or 1.0) for g, pts in truth.series.items()}
    n_truth = sum(len(pts) for pts in truth.series.values())
    pred_names = list(pred.series)
    res: dict[str, Any] = {"truth_cells": n_truth, "pred_cells": sum(len(v) for v in pred.series.values()), "extracted": bool(pred_names)}
    if not pred_names:
        for t in (0.05, 0.10):
            res[f"recall@{int(t * 100)}"] = 0.0
            res[f"precision@{int(t * 100)}"] = 0.0
            res[f"f1@{int(t * 100)}"] = 0.0
        return res
    cand: dict[tuple[str, str], list[float | None]] = {(g, p): _values_for(truth.series[g], pred, p) for g in gt_series for p in pred_names}

    def matches(g: str, p: str, t: float) -> int:
        return sum(1 for (_, tv), pv in zip(truth.series[g], cand[(g, p)], strict=True) if pv is not None and _within(pv, tv, t, floors[g]))

    # Assign predicted series to ground-truth series to maximise matches (brute force when small, else greedy).
    best: dict[str, str] = {}
    if len(gt_series) <= 5 and len(pred_names) <= 5:
        best_total = -1
        k = min(len(gt_series), len(pred_names))
        for gs in itertools.permutations(gt_series, k):
            for ps in itertools.permutations(pred_names, k):
                total = sum(matches(g, p, tol) for g, p in zip(gs, ps, strict=True))
                if total > best_total:
                    best_total, best = total, dict(zip(gs, ps, strict=True))
            if best_total == n_truth:
                break
    else:
        used: set[str] = set()
        for g in gt_series:
            options = [p for p in pred_names if p not in used]
            if not options:
                break
            p = max(options, key=lambda q: matches(g, q, tol))
            best[g] = p
            used.add(p)
    offered = sum(sum(v is not None for v in cand[(g, p)]) for g, p in best.items())
    avg_rows = n_truth / max(1, len(gt_series))
    extra = sum(min(len(pred.series[p]), avg_rows) for p in pred_names if p not in best.values())
    n_offered = offered + extra
    res["pred_cells_at_truth_positions"] = n_offered
    for t in (0.05, 0.10):
        hit = sum(matches(g, p, t) for g, p in best.items())
        recall = hit / n_truth if n_truth else 0.0
        precision = hit / n_offered if n_offered else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        res[f"recall@{int(t * 100)}"] = round(recall, 4)
        res[f"precision@{int(t * 100)}"] = round(precision, 4)
        res[f"f1@{int(t * 100)}"] = round(f1, 4)
    return res


# =========================================================================== run + report
@dataclass
class Result:
    sample_id: str
    chart_type: str
    library: str
    reader: str
    ms: float
    error: str
    metrics: dict[str, Any]


def run(samples: list[Sample], readers: list[str], client: ChartModelClient | None, progress: bool = True) -> list[Result]:
    out: list[Result] = []
    for i, s in enumerate(samples, 1):
        for r in readers:
            pred = read_pixel(s.image) if r == "pixel" else read_model(s.image, client) if client is not None else Prediction(error="no model URL")
            out.append(Result(s.id, s.chart_type, s.library, r, round(pred.ms, 1), pred.error, score(s.truth, pred)))
        if progress and (i % 10 == 0 or i == len(samples)):
            print(f"  {i}/{len(samples)} charts evaluated", flush=True)
    return out


def summarize(results: list[Result]) -> dict[str, Any]:
    def agg(rs: list[Result]) -> dict[str, Any]:
        if not rs:
            return {}
        m = lambda k: round(float(np.mean([r.metrics.get(k, 0.0) for r in rs])), 4)  # noqa: E731
        return {
            "charts": len(rs),
            "extraction_rate": round(float(np.mean([r.metrics["extracted"] for r in rs])), 4),
            "recall@5": m("recall@5"),
            "precision@5": m("precision@5"),
            "f1@5": m("f1@5"),
            "f1@10": m("f1@10"),
            "mean_ms": round(float(np.mean([r.ms for r in rs])), 1),
        }

    by_reader: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_reader[r.reader].append(r)
    out: dict[str, Any] = {"overall": {}, "by_chart_type": {}, "by_library": {}}
    for reader, rs in by_reader.items():
        out["overall"][reader] = agg(rs)
        types: dict[str, list[Result]] = defaultdict(list)
        libs: dict[str, list[Result]] = defaultdict(list)
        for r in rs:
            types[r.chart_type].append(r)
            libs[r.library].append(r)
        out["by_chart_type"][reader] = {k: agg(v) for k, v in sorted(types.items())}
        out["by_library"][reader] = {k: agg(v) for k, v in sorted(libs.items())}
    return out


def priors_from(summary: dict[str, Any], min_charts: int = 5) -> dict[str, Any]:
    """Per reader and chart type: measured F1@5 (only where enough charts were seen)."""
    pri: dict[str, Any] = {}
    for reader, types in summary["by_chart_type"].items():
        pri[reader] = {"_overall": summary["overall"][reader]["f1@5"]}
        for ct, a in types.items():
            if a.get("charts", 0) >= min_charts:
                pri[reader][ct] = a["f1@5"]
    return pri


def write_report(out_dir: Path, results: list[Result], summary: dict[str, Any], meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2), encoding="utf-8")
    (out_dir / "priors.json").write_text(json.dumps({"meta": meta, "priors": priors_from(summary)}, indent=2), encoding="utf-8")
    import csv

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["truth_cells", "pred_cells", "pred_cells_at_truth_positions", "extracted", "recall@5", "precision@5", "f1@5", "recall@10", "precision@10", "f1@10"]
        w.writerow(["sample_id", "chart_type", "library", "reader", "ms", "error", *keys])
        for r in results:
            w.writerow([r.sample_id, r.chart_type, r.library, r.reader, r.ms, r.error, *[r.metrics.get(k) for k in keys]])
    lines = [
        "# ChartNet benchmark",
        "",
        f"Run {meta['finished_at']} · dataset `{meta['dataset']}` subset `{meta['subset']}` · {meta['samples']} charts · readers: {', '.join(meta['readers'])}",
        "",
    ]
    if meta.get("license_note"):
        lines += [f"> {meta['license_note']}", ""]
    lines += ["## Overall", "", "| Reader | Charts | Extraction rate | Recall@5% | Precision@5% | F1@5% | F1@10% | Mean time |", "|---|---|---|---|---|---|---|---|"]
    for reader, a in summary["overall"].items():
        lines.append(f"| {reader} | {a['charts']} | {a['extraction_rate']:.0%} | {a['recall@5']:.0%} | {a['precision@5']:.0%} | {a['f1@5']:.0%} | {a['f1@10']:.0%} | {a['mean_ms'] / 1000:.1f} s |")
    lines += ["", "## By chart type (F1@5%)", ""]
    readers = list(summary["by_chart_type"])
    types = sorted({t for r in readers for t in summary["by_chart_type"][r]})
    lines += ["| Chart type | " + " | ".join(readers) + " | Charts |", "|---" * (len(readers) + 2) + "|"]
    for t in types:
        vals = [summary["by_chart_type"][r].get(t, {}) for r in readers]
        lines.append(f"| {t} | " + " | ".join(f"{v['f1@5']:.0%}" if v else "—" for v in vals) + f" | {max(v.get('charts', 0) for v in vals)} |")
    lines += [
        "",
        "## How to read this",
        "",
        "* A value counts as correct when it is within 5 % (or 10 %) of the ChartNet ground truth; values near zero use 1 % of that series' largest value as the tolerance.",
        "* Precision counts the reader's values at the ground-truth positions, plus a penalty for series that are not in the ground truth.",
        "* The pixel reader only reads coloured line/area/bar series with numeric axis labels; pie, heatmap, box, violin, scatter-style charts are outside its design and will score low.",
        "* `priors.json` stores these per-type scores; the app uses them to set the confidence of each reader's live values.",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    from config import load_settings

    ap = argparse.ArgumentParser(description="Evaluate chart readers on IBM ChartNet (free).")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--subset", default="core_permissive", help="core_permissive (CDLA-Permissive-2.0, default) | human_verified | core")
    ap.add_argument("--split", default=None, help="default: train (test for human_verified)")
    ap.add_argument("--parquet", nargs="*", help="Use already-downloaded ChartNet parquet file(s) instead of streaming")
    ap.add_argument("--readers", default="pixel", help="Comma list: pixel,model")
    ap.add_argument("--model-url", default=None, help="Chart-model server URL (default from app settings)")
    ap.add_argument("--chart-types", default="", help="Optional comma list to restrict chart types, e.g. line,bar,area")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output folder (default: <app data dir>/benchmarks/<timestamp>)")
    args = ap.parse_args(argv)
    settings = load_settings()
    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    bad = set(readers) - {"pixel", "model"}
    if bad:
        ap.error(f"Unknown reader(s): {', '.join(bad)}")
    split = args.split or ("test" if args.subset == "human_verified" else "train")
    client = ChartModelClient(args.model_url or settings.chart_model_url, timeout=settings.chart_model_timeout_s) if "model" in readers else None
    if client is not None and not client.health().get("ok"):
        print(f"Chart-model server not reachable at {client.base_url}; start it first (chart_model_server/README.md) or use --readers pixel.")
        return 2
    print(f"Loading {args.samples} ChartNet charts ({args.subset}/{split})…", flush=True)
    samples = load_samples(args.samples, args.subset, split, args.parquet, args.seed, {c.strip() for c in args.chart_types.split(",") if c.strip()} or None)
    if not samples:
        print("No usable samples were loaded.")
        return 1
    print(f"Evaluating {len(samples)} charts with {', '.join(readers)}…", flush=True)
    results = run(samples, readers, client)
    summary = summarize(results)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else settings.data_dir / "benchmarks" / stamp
    meta = {
        "dataset": DATASET,
        "subset": args.subset,
        "split": split,
        "samples": len(samples),
        "readers": readers,
        "model": client.health().get("model") if client else None,
        "finished_at": stamp,
        "license_note": "" if args.subset in PERMISSIVE else NONCOMMERCIAL_NOTE,
    }
    path = write_report(out_dir, results, summary, meta)
    print(f"\nReport: {path}")
    for reader, a in summary["overall"].items():
        print(f"  {reader:6s} F1@5% {a['f1@5']:.1%}  F1@10% {a['f1@10']:.1%}  extraction {a['extraction_rate']:.0%}  ({a['charts']} charts, {a['mean_ms'] / 1000:.1f} s/chart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Sample", "Prediction", "load_samples", "read_pixel", "read_model", "score", "run", "summarize", "priors_from", "write_report", "main", "asdict"]
