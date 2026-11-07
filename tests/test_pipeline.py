"""End-to-end tests of the RAG∞-style live pipeline with a scripted router (no keys, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.evidence import EvidenceRegistry, SourceRef
from agents.pipeline import LIMITER, answer_live
from agents.references import ConversationState
from agents.tools import ToolCallRecord
from agents.verifier import check, matches
from config import Settings
from core.runtime import Runtime
from ingest.files import load_dataset
from llm.catalog import split
from ops import answer_cache
from storage.db import Database
from tests.conftest import FakeScreen
from tests.fakes import FakeRouter, cite_first_number
from tests.synthetic import line_chart
from vision.ocr import ocr_status

needs_ocr = pytest.mark.skipif(not ocr_status()[0], reason="Tesseract not installed")


def _csv(n=120, seed=1) -> bytes:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-11-01 10:00", periods=n, freq="10s")
    cur = 50 + np.cumsum(rng.normal(0, 0.3, n))
    volt = 230 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"timestamp": t, "current_a": cur, "voltage_v": volt}).to_csv(index=False).encode()


@pytest.fixture(autouse=True)
def _fresh():
    answer_cache.answers(256, 600).clear()
    LIMITER._hits.clear()


def _rt(router, tmp_path, **kw):
    s = Settings(data_dir=tmp_path, **kw).validate()
    rt = Runtime(s, source_factory=lambda: FakeScreen(size=(800, 500)), router=router, db=Database(":memory:"))
    rt.service.ingest_dataset(load_dataset("plant.csv", _csv()))
    return rt, s


def test_catalog_split_handles_ollama_tags():
    assert split("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")
    assert split("google/gemma-4-31b-it:free") == ("openrouter", "google/gemma-4-31b-it:free")


def test_text_route_is_grounded_and_cited(tmp_path):
    router = FakeRouter()
    rt, s = _rt(router, tmp_path)
    st = ConversationState("c1")
    r = answer_live("What is the average of current_a?", rt, router, st, s)
    assert r.route == "text" and r.used_llm and r.verdict in ("verified", "partial")
    assert r.sources and r.sources[0]["kind"] in ("STATS", "METRICS")
    assert all(not c["images"] for c in router.calls)  # text never carries images
    assert any(e["tool"] == "calculate_statistics" for e in r.evidence)


def test_invented_number_triggers_one_regeneration(tmp_path):
    answers = iter(["The mean current is 999.25 A [REF-1].", None])

    def script(role, messages, kw):
        a = next(answers)
        return a if a else cite_first_number(messages[:2])

    router = FakeRouter(script)
    rt, s = _rt(router, tmp_path)
    r = answer_live("What is the average of current_a?", rt, router, ConversationState("c"), s)
    assert r.regenerated and "999.25" not in r.answer and r.verdict != "unverified"


def test_no_model_gives_computed_summary(tmp_path):
    router = FakeRouter()
    router.enabled = False
    rt, s = _rt(router, tmp_path)
    r = answer_live("What is the average of current_a?", rt, router, ConversationState("c"), s)
    assert r.kind == "offline" and "Automatic summary" in r.answer and not router.calls


def test_cache_respects_data_version(tmp_path):
    router = FakeRouter()
    rt, s = _rt(router, tmp_path)
    q = "What is the maximum of voltage_v?"
    answer_live(q, rt, router, ConversationState("a"), s)
    assert answer_live(q, rt, router, ConversationState("b"), s).cached
    rt.service.ingest_dataset(load_dataset("more.csv", _csv(seed=7)))  # new data -> new version
    assert not answer_live(q, rt, router, ConversationState("c"), s).cached


def test_agent_mode_runs_master_tools(tmp_path):
    seen = {"n": 0}

    def script(role, messages, kw):
        seen["n"] += 1
        if seen["n"] == 1 and kw["tools"]:
            return ("tool", "detect_anomalies", {"metric": "current_a"})
        return cite_first_number(messages[:2])

    router = FakeRouter(script)
    rt, s = _rt(router, tmp_path)
    r = answer_live("Why did current_a increase?", rt, router, ConversationState("c"), s)
    assert any(st.kind == "action" and st.tool == "detect_anomalies" for st in r.steps)
    names = [t["function"]["name"] for t in router.calls[0]["tools"]]
    assert "record_recommendation" not in names  # side-effect tools are not given to the model


def test_rate_limit(tmp_path):
    router = FakeRouter()
    rt, s = _rt(router, tmp_path, rate_limit_per_min=1)
    answer_live("What is the average of current_a?", rt, router, ConversationState("c"), s, session_key="k")
    assert answer_live("What is the max of current_a?", rt, router, ConversationState("c"), s, session_key="k").kind == "rate_limited"


@needs_ocr
@pytest.mark.parametrize("fail_vision", [False, True])
def test_screen_question_uses_vision_with_crops(tmp_path, fail_vision):
    x = np.arange(60)
    img, _, _ = line_chart(series={"Bus Voltage": 230 + 5 * np.sin(x / 6)}, x=x, title="Bus Voltage (V)", ylim=(220, 240))
    router = FakeRouter(fail_vision=fail_vision)
    s = Settings(data_dir=tmp_path).validate()
    rt = Runtime(s, source_factory=lambda: FakeScreen(provider=lambda n: img, size=(800, 500)), router=router, db=Database(":memory:"))
    rt.manual_capture()
    r = answer_live("What do you see on my screen?", rt, router, ConversationState("v"), s)
    vision_calls = [c for c in router.calls if c["role"] == "vision"]
    assert vision_calls and vision_calls[0]["images"]
    assert all(not c["images"] for c in router.calls if c["role"] != "vision")
    if fail_vision:
        assert r.route == "text" and any("vision model" in w for w in r.warnings)
    else:
        assert r.route == "vision" and r.images_sent >= 1


def test_verifier_rounding_and_image_cap():
    assert matches("12.3", 12.3, [12.3456]) and not matches("12.4", 12.4, [12.3456])
    reg = EvidenceRegistry()
    reg.add_record(ToolCallRecord("calculate_statistics", {"metric": "x"}, {"mean": 12.3456}, True, 1))
    img = reg.add_note("SCREEN-IMAGE", "chart1", "image", data={}, visual=True)
    ok = check("Mean 12.35 [REF-1].", reg.refs, "q")
    assert ok.verdict == "verified"
    vis = check(f"The peak looks like 17.5 {img.label}.", reg.refs, "q")
    assert vis.grounded and vis.verdict == "partial" and vis.image_values == 1
    bad = check("Mean 55.5 [REF-1].", reg.refs, "q")
    assert not bad.grounded and bad.unverified_numbers == ["55.5"]
    assert isinstance(SourceRef.from_dict(reg.refs[0].as_dict()), SourceRef)


READING = ('{"summary":"Bus voltage oscillates around 230 V.","charts":[{"title":"Bus Voltage (V)","type":"line",'
           '"x_axis":"time","y_axis":"V 220-240","series":[{"name":"Bus Voltage","unit":"V","latest_value":233.5,'
           '"min_value":225.1,"max_value":235.0,"trend":"oscillating","notes":"period ~50 samples"}],'
           '"thresholds":"none","anomalies":"none"}],"kpis":[{"label":"Voltage","value":233.5,"unit":"V","status":"normal"}],'
           '"indicators":[],"findings":[{"severity":"info","finding":"steady oscillation","evidence":"sine shape"}],'
           '"assessment":"healthy","readability":"good"}')


def test_vision_first_reading_and_inference(tmp_path):
    from PIL import Image

    def script(role, messages, kw):
        sys_txt = messages[0]["content"]
        if role == "vision" and "Answer with JSON only" in sys_txt:
            return READING
        return "The bus voltage is oscillating, latest about 233.5 V [REF-1]."

    router = FakeRouter(script)
    s = Settings(data_dir=tmp_path).validate()
    rt = Runtime(s, source_factory=lambda: FakeScreen(size=(800, 500)), router=router, db=Database(":memory:"))
    notes = rt.analyze_image_now(Image.new("RGB", (900, 600), "white"), "upload: test.png")
    assert notes["charts"][0]["series"][0]["trend"] == "oscillating"
    assert "Voltage" in rt.service.metric_names()  # vision-read KPI became an observation
    r = answer_live("Explain this chart", rt, router, ConversationState("v"), s)
    assert r.route == "vision" and r.images_sent >= 1
    kinds = [e["kind"] for e in r.evidence]
    assert "VISION-READING" in kinds and "SCREEN-IMAGE" in kinds
    assert r.verdict in ("verified", "partial") and "233.5" in r.answer
