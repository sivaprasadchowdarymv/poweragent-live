import io

import numpy as np
import pandas as pd
import pytest

from ingest.files import (
    FileValidationError,
    load_dataset,
    normalize_dataset,
    parse_timestamps,
    split_unit,
    to_numeric_clean,
    validate_upload,
)
from analysis.reconcile import compare_sources
from analysis.stats import prepare_series
from config import ConfigError, Settings, load_settings
from core.services import DataService
from storage.db import metric_key, redact

CSV = b"Time,Voltage (V),Current [A],note\n2026-01-01 00:00:00,230,5,a\n2026-01-01 00:00:02,231.5,5.2,b\n2026-01-01 00:00:04,x,5.1,c\n2026-01-01 00:00:04,229,,d\n,228,5,e\n"


def test_validate_upload_rejects_bad_files():
    with pytest.raises(FileValidationError, match="not supported"):
        validate_upload("evil.exe", b"MZ")
    with pytest.raises(FileValidationError, match="empty"):
        validate_upload("a.csv", b"")
    with pytest.raises(FileValidationError, match="not a valid Excel"):
        validate_upload("a.xlsx", b"hello")
    with pytest.raises(FileValidationError, match="binary"):
        validate_upload("a.csv", b"a,b\n\x00\x01")
    with pytest.raises(FileValidationError, match="limit"):
        validate_upload("a.csv", b"a" * 2_000_000, max_mb=1)
    assert validate_upload("../../dir/ok.CSV", b"a,b\n1,2") == ".csv"


def test_csv_profile_units_and_quality_report():
    ds = load_dataset("../../t.csv", CSV)
    assert ds.name == "t.csv"  # path components stripped
    assert ds.timestamp_col == "Time"
    assert set(ds.metric_cols) == {"Voltage (V)", "Current [A]"}
    assert ds.units == {"Voltage (V)": "V", "Current [A]": "A"}
    obs, rep = normalize_dataset(ds)
    assert rep["non_numeric_cells"]["Voltage (V)"] == [2]
    assert rep["missing_cells"]["Current [A]"] == 1
    assert rep["invalid_timestamps"] == [4]
    assert rep["duplicate_timestamps"] >= 1
    assert 0 < rep["quality_score"] < 1
    v = obs[obs.metric == "Voltage"]
    assert v["value"].tolist() == [230.0, 231.5, 229.0]
    assert set(obs["status"]) == {"verified"} and set(obs["source_type"]) == {"file"}
    assert v["meta"].iloc[0]["row"] == 0 and v["meta"].iloc[0]["column"] == "Voltage (V)"


def test_xlsx_roundtrip_and_no_timestamp_column():
    df = pd.DataFrame({"Load (kW)": [10.0, 12.5, 11.0], "Freq (Hz)": [50.0, 49.9, 50.1]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    ds = load_dataset("plant.xlsx", buf.getvalue())
    assert ds.timestamp_col is None
    obs, rep = normalize_dataset(ds)
    assert rep["time_basis"] == "row_index"
    assert obs[obs.metric == "Load"]["value"].tolist() == [10.0, 12.5, 11.0]


def test_timestamp_parsing_tz_and_epoch():
    s = parse_timestamps(pd.Series(["2026-01-01 05:30:00"]), tz="Asia/Kolkata")
    assert s.iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    ms = parse_timestamps(pd.Series([1767225600000, 1767225601000]))
    assert ms.iloc[1] - ms.iloc[0] == pd.Timedelta(seconds=1)
    sec = parse_timestamps(pd.Series([1767225600, 1767225660]))
    assert sec.iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")


def test_numeric_cleaning_and_units():
    assert to_numeric_clean(pd.Series(["1,200", "45%", "230 V", "abc"])).tolist()[:3] == [1200.0, 45.0, 230.0]
    assert np.isnan(to_numeric_clean(pd.Series(["abc"])).iloc[0])
    assert split_unit("Power [kW]") == ("Power", "kW")
    assert split_unit("Voltage") == ("Voltage", "")


def test_storage_roundtrip_exact_timestamps(db):
    svc = DataService(db)
    ds = load_dataset("t.csv", CSV)
    rep = svc.ingest_dataset(ds)
    assert rep["stored"] == 6
    sd, win, meta = svc.get_series("voltage", "all")
    assert sd.series.index[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert sd.series.index[-1] == pd.Timestamp("2026-01-01T00:00:04Z")
    assert meta["source_id"] == ds.id
    # re-ingest replaces instead of duplicating
    svc.ingest_dataset(ds)
    assert db.count_observations() == 6
    assert svc.resolve_metric("VOLTAGE (V)") == "Voltage"
    assert svc.resolve_metric("voltge") == "Voltage"
    assert svc.resolve_metric("pressure") is None


def test_correction_and_deletion(db):
    svc = DataService(db)
    svc.ingest_dataset(load_dataset("t.csv", CSV))
    obs = db.query_observations(metric="Voltage")
    db.correct_observation(int(obs.iloc[0]["id"]), 999.0, "fixed")
    after = db.query_observations(metric="Voltage")
    assert after.iloc[0]["value"] == 999.0 and after.iloc[0]["status"] == "corrected"
    assert db.corrections()[0]["old_value"] == 230.0
    assert db.delete_all_measurements() > 0 and db.count_observations() == 0


def test_metric_key_and_redaction():
    assert metric_key("Voltage (V)") == metric_key("voltage") == "voltage"
    assert metric_key("Phase-A  Current") == "phase a current"
    r = redact("login bob@example.com password: hunter2 card 4111 1111 1111 1111")
    assert "bob@" not in r and "hunter2" not in r and "4111" not in r


def test_reconcile_reports_discrepancies(ts_index):
    idx = ts_index(20)
    a = prepare_series(pd.Series(np.full(20, 230.0), index=idx), "Voltage", "V", "file", "verified")
    vals = np.full(20, 230.5)
    vals[5] = 250
    b = prepare_series(pd.Series(vals, index=idx + pd.Timedelta(milliseconds=500)), "Voltage", "V", "screen", "estimated")
    r = compare_sources(a, b)
    assert r["paired_samples"] == 20
    assert r["discrepancies"] == 1
    assert r["examples"][0]["b"] == 250
    assert "not merged" in r["policy"]
    c = compare_sources(a, prepare_series(pd.Series(np.full(5, 1.0), index=ts_index(5, start="2030-01-01")), "Voltage", "kV"))
    assert c["paired_samples"] == 0 and c["unit_mismatch"]


def test_reconcile_mixed_datetime_resolutions():
    """Regression: file data parsed at second resolution vs screen data at ns."""
    idx_s = pd.date_range("2026-01-01", periods=10, freq="2s", tz="UTC").as_unit("s")
    idx_ns = (pd.date_range("2026-01-01", periods=10, freq="2s", tz="UTC") + pd.Timedelta(milliseconds=200)).as_unit("ns")
    a = prepare_series(pd.Series(np.arange(10.0), index=idx_s), "V")
    b = prepare_series(pd.Series(np.arange(10.0) + 0.01, index=idx_ns), "V")
    assert compare_sources(a, b)["paired_samples"] == 10


def test_screen_and_file_sources_compared_through_service(db):
    from agents.tools import ToolContext, t_compare_sources
    from analysis.alerts import AlertEngine

    svc = DataService(db)
    t = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("s"), periods=30, freq="2s")
    csv = pd.DataFrame({"Timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "Feeder Voltage (V)": np.full(30, 230.0)}).to_csv(index=False).encode()
    svc.ingest_dataset(load_dataset("f.csv", csv))
    rows = [{"timestamp": ts + pd.Timedelta(milliseconds=300), "source_type": "screen", "source_id": "screen_live", "metric": "Feeder Voltage", "value": 230.4 if i != 7 else 245.0, "unit": "V", "confidence": 0.9, "status": "estimated", "meta": {}} for i, ts in enumerate(t)]
    db.add_observations(pd.DataFrame(rows))
    res = t_compare_sources(ToolContext(svc, AlertEngine(db)), "feeder voltage")
    assert res["paired_samples"] == 30 and res["discrepancies"] == 1
    # verified file data is preferred automatically; the estimate is never merged in
    sd, _, meta = svc.get_series("Feeder Voltage", "all")
    assert sd.status == "verified" and meta["other_sources"] == ["screen_live"]


def test_settings_validation():
    assert Settings().validate().capture_interval_s == 2.0
    with pytest.raises(ConfigError):
        Settings(capture_interval_s=0.01).validate()
    s = load_settings({"CAPTURE_INTERVAL": "1.5", "VISION_NOTES": "true", "MODEL_PROFILE": "quality"})
    assert s.capture_interval_s == 1.5 and s.vision_enabled and s.models_for("vision")[0].startswith("openrouter:qwen")
    with pytest.raises(ConfigError):
        load_settings({"RETENTION_DAYS": "many"})
    local = load_settings({"LOCAL_ONLY": "true", "GROQ_API_KEY": "x"})
    assert not local.has_groq and all(e.startswith("ollama:") for e in local.models_for("master"))
