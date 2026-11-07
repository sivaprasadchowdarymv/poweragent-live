"""Chart-model integration (Granite Vision / ChartAst): server HTTP layer with a
stub engine (no GPU or model weights here), client,
Markdown-table parsing, cross-check against the pixel reader, storage as a
separate source, extraction scheduling and chat tools."""

import base64
import importlib.util
import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from agents.references import ConversationState
from agents.tools import ToolContext, t_ask_chart_model, t_read_chart_table
from analysis.alerts import AlertEngine
from config import ConfigError, Settings
from core.models import utcnow
from core.runtime import Runtime
from core.services import CHART_MODEL_SOURCE_ID, DataService
from storage.db import Database
from vision.chart_model import ChartModelClient, ChartModelError, cross_check, parse_chart_table, parse_markdown_table, read_frame
from vision.chart_reader import analyze_chart_image
from vision.ocr import ocr_status, read_line, run_ocr

from tests.conftest import FakeScreen
from agents.pipeline import answer_live
from tests.fakes import FakeRouter
from tests.synthetic import bar_chart, line_chart

needs_ocr = pytest.mark.skipif(not ocr_status()[0], reason="Tesseract not installed")
SERVER_DIR = Path(__file__).resolve().parents[1] / "chart_model_server"


def load_module(name="common"):
    import sys

    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    spec = importlib.util.spec_from_file_location(f"cms_{name}", SERVER_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_server_module():
    return load_module("common")


class StubEngine:
    """Stands in for the model: records requests, returns canned text (default:
    Granite-style CSV)."""

    name = "stub-granite"
    output_format = "csv"
    info = {"model": "stub-granite", "quant": "none", "device": "cpu-stub", "gpu_memory_gb": 0}

    def __init__(self, table_text="Year,Sales\n2019,12", delay=0.0):
        self.table_text = table_text
        self.delay = delay
        self.calls = []

    def run(self, task, image, question, max_new_tokens, temperature):
        self.calls.append({"task": task, "question": question, "size": image.size, "max_new": max_new_tokens, "temperature": temperature})
        if self.delay:
            time.sleep(self.delay)
        if task == "table":
            return self.table_text
        return "The highest value is 41."


@pytest.fixture
def server():
    mod = load_server_module()
    eng = StubEngine()
    srv = mod.ChartModelHTTPServer(("127.0.0.1", 0), eng, busy_timeout_s=0.3)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    srv.engine_stub = eng
    srv.mod = mod
    yield srv
    srv.shutdown()
    srv.server_close()


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ server (real HTTP code, stub model)
def test_chartast_prompts_match_repo():
    m = load_module("chartast_server")
    assert m.build_prompt("table") == (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nconvert this chart to a table.\n\n### Response:"
    )
    assert "Please answer my question based on the chart: What is max?" in m.build_prompt("qa", "What is max?")
    assert m.clean_answer("| a | b |\n### Instruction: x") == "| a | b |"
    assert m.ChartAstEngine.output_format == "markdown"


def test_granite_prompts_match_model_card():
    g = load_module("granite_server")
    assert g.build_messages("table", "") == [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<chart2csv>"}]}]
    assert g.build_messages("summary", "")[0]["content"][1]["text"] == "<chart2summary>"
    assert g.build_messages("qa", " Which is highest? ")[0]["content"][1]["text"] == "Which is highest?"
    assert g.clean_output("```csv\nYear,Sales\n2019,12\n```") == "Year,Sales\n2019,12"
    assert g.clean_output("Year,Sales\n2019,12") == "Year,Sales\n2019,12"
    assert g.GraniteVisionEngine.output_format == "csv" and g.DEFAULT_MODEL == "ibm-granite/granite-vision-4.1-4b"


def test_server_roundtrip(server):
    img = Image.new("RGB", (640, 400), "white")
    r = requests.post(server.url + "/v1/chart_to_table", json={"image": png_b64(img)}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Year,Sales\n2019,12" and body["format"] == "csv" and body["model"] == "stub-granite"
    call = server.engine_stub.calls[-1]
    assert call["task"] == "table" and call["size"] == (640, 400) and call["temperature"] == 0.0
    h = requests.get(server.url + "/health", timeout=5).json()
    assert h["ok"] and h["requests"] == 1 and h["output_format"] == "csv"


def test_server_validation_and_busy(server):
    url = server.url
    assert requests.post(url + "/v1/chart_qa", json={"image": png_b64(Image.new("RGB", (10, 10)))}, timeout=5).status_code == 400  # no question
    assert requests.post(url + "/v1/chart_to_table", json={"image": "not-base64!!"}, timeout=5).status_code == 400
    assert requests.post(url + "/v1/unknown", json={}, timeout=5).status_code == 404
    server.gpu_lock.acquire()
    try:
        r = requests.post(url + "/v1/chart_to_table", json={"image": png_b64(Image.new("RGB", (10, 10)))}, timeout=5)
        assert r.status_code == 503
    finally:
        server.gpu_lock.release()


def test_client_kinds(server):
    c = ChartModelClient(server.url, timeout=5)
    assert c.health()["ok"]
    assert "12" in c.chart_to_table(Image.new("RGB", (50, 50)))["text"]
    assert "41" in c.chart_qa(Image.new("RGB", (50, 50)), "max?")["text"]
    off = ChartModelClient("http://127.0.0.1:9", timeout=2, connect_timeout=0.5)
    assert off.health()["ok"] is False
    with pytest.raises(ChartModelError) as e:
        off.chart_to_table(Image.new("RGB", (5, 5)))
    assert e.value.kind == "offline"


# ------------------------------------------------------------------ parsing
def test_parse_granite_csv_variants():
    t = parse_chart_table("```csv\nMonth,Load (kW),Voltage\nJan,12.5,230\nFeb,\"1,240\",231\n```")
    assert t.ok and t.series["Load (kW)"] == [("Jan", 12.5), ("Feb", 1240.0)] and t.series["Voltage"][-1] == ("Feb", 231.0)
    semi = parse_chart_table("Year;Value\n2020;3,5\n2021;4")
    assert semi.series["Value"][-1] == ("2021", 4.0)
    tab = parse_chart_table("x\ty\n1\t2\n2\t4")
    assert tab.series["y"] == [("1", 2.0), ("2", 4.0)] and tab.x_kind == "number"
    assert not parse_chart_table("This chart shows sales rising, then falling.").ok


def test_parse_markdown_pipe_table():
    t = parse_markdown_table("Here is the table:\n| Year | Phase A | Phase B |\n|:---|---:|---:|\n| 2019 | 230.5 | 221 |\n| 2020 | 1,240 | 45% |\n| 2021 | n/a | $3.5 |")
    assert t.ok and t.x_kind == "number"
    assert t.series["Phase A"] == [("2019", 230.5), ("2020", 1240.0)]
    assert t.series["Phase B"] == [("2019", 221.0), ("2020", 45.0), ("2021", 3.5)]


def test_parse_linearised_title_and_no_outer_pipes():
    t = parse_markdown_table("TITLE | Feeder Load <0x0A> Feeder | Load (kW) <0x0A> F1 | 12 <0x0A> F2 | 30.5")
    assert t.title == "Feeder Load"
    assert t.series["Load (kW)"] == [("F1", 12.0), ("F2", 30.5)]
    assert t.x_kind == "category"


def test_parse_transposed_and_time_and_garbage():
    t = parse_markdown_table("| Metric | 2019 | 2020 | 2021 |\n| Revenue | 10 | 12 | 15 |\n| Cost | 7 | 8 | 9 |")
    assert t.series["Revenue"][-1] == ("2021", 15.0) and "transposed" in " ".join(t.parse_notes)
    tt = parse_markdown_table("| Time | Voltage |\n| 10:00 | 230 |\n| 10:05 | 231 |")
    assert tt.x_kind == "time"
    bad = parse_markdown_table("I cannot read this chart.")
    assert not bad.ok and bad.parse_notes


# ------------------------------------------------------------------ cross-check against the pixel reader
def _pixel(img):
    return analyze_chart_image(img, run_ocr(img), line_reader=lambda b: read_line(img, b))[0][0]


@needs_ocr
def test_cross_check_agrees_with_true_table_and_flags_wrong_one():
    x = np.arange(60)
    y = 230 + 10 * np.sin(x / 8.0)
    img, _, _ = line_chart(series={"Voltage": y}, x=x, title="Bus Voltage (V)")
    chart = _pixel(img)
    rows = "\n".join(f"| {i} | {y[i]:.2f} |" for i in range(0, 60, 5))
    good = cross_check(parse_markdown_table("| x | Voltage |\n|---|---|\n" + rows), chart)
    assert good.status == "agree" and good.compared_points >= 10 and good.error_pct_of_axis < 1.5
    wrong_rows = "\n".join(f"| {i} | {y[i] + 15:.2f} |" for i in range(0, 60, 5))
    bad = cross_check(parse_markdown_table("| x | Voltage |\n|---|---|\n" + wrong_rows), chart)
    assert bad.status == "disagree"


@needs_ocr
def test_cross_check_bar_chart():
    img, values = bar_chart()
    chart = _pixel(img)
    table = parse_markdown_table("| Feeder | Load |\n|---|---|\n" + "\n".join(f"| F{i + 1} | {v} |" for i, v in enumerate(values)))
    assert cross_check(table, chart).status == "agree"


# ------------------------------------------------------------------ storage, source choice, tools
def _true_table_engine():
    x = np.arange(60)
    y = 220 + 0.2 * x
    y[-1] = 236.0
    csv_rows = "\n".join(f"{i},{y[i]:.2f}" for i in list(range(0, 60, 5)) + [59])
    return StubEngine("```csv\nx,Bus Voltage\n" + csv_rows + "\n```"), x, y


@needs_ocr
def test_readings_stored_as_separate_source_and_tools(tmp_path):
    mod = load_server_module()
    eng, x, y = _true_table_engine()
    srv = mod.ChartModelHTTPServer(("127.0.0.1", 0), eng)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        chart_img, _, _ = line_chart(series={"Bus Voltage": y}, x=x, title="Bus Voltage (V)", ylim=(200, 270))
        img = Image.new("RGB", (1700, 1100), "white")  # a dashboard: the chart is one panel of a larger screen
        img.paste(chart_img, (700, 450))
        words = run_ocr(img)
        charts, kpis, _ = analyze_chart_image(img, words, line_reader=lambda b: read_line(img, b))
        readings = read_frame(ChartModelClient(url), img, charts)
        assert readings[0].check.status == "agree"
        w, h = eng.calls[0]["size"]
        assert w < img.width and h < img.height  # only the chart panel was sent, not the whole screen
        assert w >= 800 - 10 and h >= 500 - 60  # ...including its title, axes and legend area
        db = Database(":memory:")
        svc = DataService(db)
        from vision.chart_reader import ScreenAnalysis

        svc.add_screen_analysis(ScreenAnalysis(utcnow(), 1, img.size, charts, kpis, [], 0, 0))
        n = svc.add_chart_model_readings(readings, 1, utcnow())
        assert n == 1
        obs = db.query_observations(source_id=CHART_MODEL_SOURCE_ID)
        assert obs.iloc[0]["value"] == pytest.approx(236.0) and obs.iloc[0]["confidence"] == 0.9 and obs.iloc[0]["status"] == "estimated"
        assert obs.iloc[0]["unit"] == "V"
        # Two screen estimates of the same metric: the more confident one answers; nothing is merged.
        sid, cands, reason = svc.choose_source("Bus Voltage")
        assert len({c["source_id"] for c in cands}) == 2
        ctx = ToolContext(svc, AlertEngine(db))
        r = t_read_chart_table(ctx)
        assert r["available"] and r["readings"][0]["cross_check_with_pixel_reader"]["status"] == "agree"
        assert "chart_model_readings" in svc.screen_context()
        ctx.chart_model = ChartModelClient(url)
        ctx.latest_image = lambda: img
        a = t_ask_chart_model(ctx, "What is the highest value?")
        assert "41" in a["chart_model_answer"] and "estimate" in a["status"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_settings_guard_chart_model_url():
    with pytest.raises(ConfigError, match="chart-model server"):
        Settings(chart_model_url="http://192.168.1.50:8600", local_only=True).validate()
    from config import load_settings

    new = load_settings({"CHART_MODEL_ENABLED": "true", "CHART_MODEL_INTERVAL": "12"})
    assert new.chart_model_enabled and new.chart_model_interval_s == 12
    assert Settings(chart_model_url="http://192.168.1.50:8600", local_only=False).validate()


# ------------------------------------------------------------------ live pipeline + chat
@needs_ocr
def test_live_pipeline_with_chart_model_and_chat():
    mod = load_server_module()
    eng, x, y = _true_table_engine()
    eng.delay = 1.5  # slower than capture: readings must be skipped while busy, capture must continue
    srv = mod.ChartModelHTTPServer(("127.0.0.1", 0), eng)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    img, _, _ = line_chart(series={"Bus Voltage": y}, x=x, title="Bus Voltage (V)", ylim=(200, 270))
    img = img.resize((800, 500))
    if True:
        s = Settings(capture_interval_s=0.25, extraction_interval_s=0.5, dedup_threshold=0.0,
                     chart_model_enabled=True, chart_model_url=f"http://127.0.0.1:{srv.server_address[1]}", chart_model_interval_s=2).validate()
        rt = Runtime(s, source_factory=lambda: FakeScreen(provider=lambda n: img, size=(800, 500)), db=Database(":memory:"), router=FakeRouter())
        rt.start_monitoring()
        try:
            end = time.monotonic() + 40
            while time.monotonic() < end and not rt.db.query_observations(source_id=CHART_MODEL_SOURCE_ID).shape[0]:
                time.sleep(0.2)
            assert rt.service.chart_model is not None
            caps = rt.capture.metrics.captures
            time.sleep(1.0)
            assert rt.capture.metrics.captures > caps  # capture is independent of the chart model
            em = rt.extraction.metrics
            assert em.chart_model_calls >= 1 and em.chart_model_errors == 0
            state = ConversationState(rt.db.new_conversation())
            res = answer_live("What is happening on my screen?", rt, rt.router, state, s.with_updates(image_answers=False))
            assert [e["tool"] for e in res.evidence][:2] == ["get_latest_screen_context", "read_chart_table"]
            res2 = answer_live("According to the chart, what is the highest value?", rt, rt.router, state, s.with_updates(image_answers=False))
            assert any(e["tool"] == "ask_chart_model" and e["ok"] for e in res2.evidence)
        finally:
            rt.stop_monitoring()
            srv.shutdown()
            srv.server_close()
