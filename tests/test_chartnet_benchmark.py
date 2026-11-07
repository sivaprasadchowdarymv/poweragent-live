"""ChartNet benchmark harness, run end-to-end on a local parquet file in
ChartNet's format (id, image{bytes}, code, csv, summary, chart_type, library).
Real ChartNet data cannot be downloaded in the test environment, so the charts
are synthetic but rendered and scored exactly as real samples would be."""

import importlib.util
import io
import json
import sys
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metrics import chartnet_eval as ce
from metrics.priors import load_priors, prior_factor, scaled
from core.models import utcnow
from core.services import DataService
from storage.db import Database
from vision.chart_model import parse_chart_table
from vision.chart_reader import AxisCalibration, ChartEstimate, ScreenAnalysis, SeriesEstimate
from vision.ocr import ocr_status

needs_ocr = pytest.mark.skipif(not ocr_status()[0], reason="Tesseract not installed")
SERVER_DIR = Path(__file__).resolve().parents[1] / "chart_model_server"


def _png(fig, size):
    buf = io.BytesIO()
    fig.set_size_inches(size[0] / 100, size[1] / 100)
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def make_rows():
    """Six charts with ground-truth CSV; image sizes are unique so the stub
    model server can look the truth up by size."""
    rows = []
    rng = np.random.default_rng(3)
    # line charts with numeric x
    for k, size in enumerate([(800, 500), (810, 500), (820, 500)]):
        x = np.arange(2010, 2022)
        y = np.round(50 + np.cumsum(rng.normal(0, 4, len(x))), 1)
        fig, ax = plt.subplots()
        ax.plot(x, y, color="#1f77b4", lw=2.2)
        ax.set_title(f"Output {k}", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, color="#dddddd")
        fig.tight_layout()
        csv = "Year,Output\n" + "\n".join(f"{a},{b}" for a, b in zip(x, y, strict=True))
        rows.append({"id": f"line{k}", "image": {"bytes": _png(fig, size), "path": None}, "code": "plt.plot(...)", "csv": csv, "summary": "", "chart_type": "line", "library": "matplotlib"})
    # bar charts with category x
    for k, size in enumerate([(800, 510), (800, 520)]):
        cats = ["A", "B", "C", "D", "E"]
        vals = np.round(rng.uniform(10, 45, 5), 1)
        fig, ax = plt.subplots()
        ax.bar(cats, vals, color="#2ca02c", width=0.6)
        ax.set_ylim(0, 50)
        ax.set_title(f"Load {k}", fontsize=15)
        ax.tick_params(labelsize=13)
        fig.tight_layout()
        csv = "Feeder,Load\n" + "\n".join(f"{a},{b}" for a, b in zip(cats, vals, strict=True))
        rows.append({"id": f"bar{k}", "image": {"bytes": _png(fig, size), "path": None}, "code": "plt.bar(...)", "csv": csv, "summary": "", "chart_type": "bar", "library": "matplotlib"})
    # a pie chart: outside the pixel reader's design
    fig, ax = plt.subplots()
    ax.pie([30, 50, 20], labels=["x", "y", "z"])
    rows.append({"id": "pie0", "image": {"bytes": _png(fig, (700, 700)), "path": None}, "code": "plt.pie(...)", "csv": "Part,Share\nx,30\ny,50\nz,20", "summary": "", "chart_type": "pie", "library": "matplotlib"})
    return rows


@pytest.fixture
def parquet_file(tmp_path):
    rows = make_rows()
    path = tmp_path / "chartnet_like.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path, rows


def test_score_metric_basics():
    truth = parse_chart_table("x,A,B\n1,10,100\n2,20,200\n3,30,300")
    perfect = ce.Prediction(series={"A": [("1", 10), ("2", 20), ("3", 30)], "B": [("1", 100), ("2", 200), ("3", 300)]})
    assert ce.score(truth, perfect)["f1@5"] == 1.0
    swapped_names = ce.Prediction(series={"s1": [("1", 100), ("2", 200), ("3", 300)], "s2": [("1", 10.3), ("2", 20.4), ("3", 31)]})
    assert ce.score(truth, swapped_names)["f1@5"] == 1.0  # series matched by values when names differ
    off7 = ce.Prediction(series={"A": [("1", 10.7), ("2", 21.4), ("3", 32.1)]})
    s = ce.score(truth, off7)
    assert s["recall@5"] == 0.0 and s["recall@10"] == 0.5 and s["precision@10"] == 1.0
    positional = ce.Prediction(series={"A": [(None, 10), (None, 15), (None, 20), (None, 25), (None, 30)]}, positional=True, fractions={"A": [0, 0.25, 0.5, 0.75, 1]})
    assert ce.score(truth, positional)["recall@5"] == 0.5  # A fully right, B missing
    dense = ce.Prediction(series={"A": [(str(x), 10 * x) for x in np.linspace(1, 3, 300)]})  # 300 samples, all on the true line
    d = ce.score(truth, dense)
    assert d["recall@5"] == 0.5 and d["precision@5"] == 1.0  # not punished for density
    spurious = ce.Prediction(series={"A": [("1", 10), ("2", 20), ("3", 30)], "noise": [("1", 5), ("2", 5), ("3", 5)]})
    assert ce.score(truth, spurious)["precision@5"] == 0.5  # invented series are penalised
    empty = ce.score(truth, ce.Prediction())
    assert empty["f1@5"] == 0.0 and not empty["extracted"]


def test_load_samples_from_parquet(parquet_file):
    path, rows = parquet_file
    samples = ce.load_samples(10, parquet=[str(path)])
    assert len(samples) == len(rows)
    assert {s.chart_type for s in samples} == {"line", "bar", "pie"}
    assert samples[0].truth.ok and samples[0].image.size == (800, 500)


@needs_ocr
def test_end_to_end_cli_with_pixel_and_model_readers(parquet_file, tmp_path, monkeypatch):
    path, rows = parquet_file
    truth_by_size = {}
    for r in rows:
        from PIL import Image

        truth_by_size[Image.open(io.BytesIO(r["image"]["bytes"])).size] = r["csv"]
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    spec = importlib.util.spec_from_file_location("cms_common_b", SERVER_DIR / "common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    class TruthEngine:  # a "perfect" chart model: proves the scoring/plumbing, not model quality
        name, output_format = "truth-stub", "csv"
        info = {"model": "truth-stub"}

        def run(self, task, image, question, max_new_tokens, temperature):
            return "```csv\n" + truth_by_size[image.size] + "\n```"

    srv = common.ChartModelHTTPServer(("127.0.0.1", 0), TruthEngine())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "bench"
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    try:
        code = ce.main(["--parquet", str(path), "--samples", "10", "--readers", "pixel,model", "--model-url", f"http://127.0.0.1:{srv.server_address[1]}", "--out", str(out)])
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["overall"]["model"]["f1@5"] == 1.0
    px = summary["by_chart_type"]["pixel"]
    assert px["line"]["f1@5"] >= 0.8, px
    assert px["bar"]["f1@5"] >= 0.8, px
    assert px["pie"]["f1@5"] <= 0.2  # outside the pixel reader's design — reported honestly
    report = (out / "report.md").read_text()
    assert "| pixel |" in report and "| line |" in report
    assert (out / "results.csv").read_text().count("\n") == 1 + 2 * len(rows)


def test_priors_scale_live_confidence(tmp_path):
    run = tmp_path / "benchmarks" / "20260101T000000Z"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"meta": {"samples": 1, "subset": "core_permissive"}, "overall": {}, "by_chart_type": {}}))
    (run / "priors.json").write_text(json.dumps({"priors": {"pixel": {"_overall": 0.6, "line": 0.9, "bar chart": 0.5}}}))
    pri = load_priors(tmp_path)
    assert pri["_run"] == "20260101T000000Z"
    assert prior_factor(pri, "pixel", "line") == 0.9 and prior_factor(pri, "pixel", "bar") == 0.5
    assert prior_factor(pri, "pixel", "area") == 0.6 and prior_factor(pri, "model", "line") is None
    assert scaled(0.9, 0.5) == 0.45 and scaled(0.9, None) == 0.9 and scaled(0.5, 1.5) == 0.5
    svc = DataService(Database(":memory:"))
    svc.priors = pri
    ax = AxisCalibration("y", "linear", 0, 1, False, [(0, 0), (100, 100)], 2, 0, 1.0)
    chart = ChartEstimate("chart1", (0, 0, 100, 100), "Voltage", "line", None, ax, [SeriesEstimate("Voltage", "#000", "line", [0, 1], [1, 2], True, 2.0, 1.0, 2.0, False, 0.8)], "V", 0.8)
    svc.add_screen_analysis(ScreenAnalysis(utcnow(), 1, (100, 100), [chart], [], [], 0, 0))
    obs = svc.db.query_observations(metric="Voltage")
    assert obs.iloc[0]["confidence"] == pytest.approx(0.72) and obs.iloc[0]["meta"]["benchmark_f1"] == 0.9
