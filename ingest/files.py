"""CSV/XLSX ingestion: validation, profiling, timestamp parsing, normalization.

The uploaded bytes are never modified or written anywhere; only a SHA-256
digest, the file name and the parsed data are kept.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.models import OBSERVATION_COLUMNS, SourceType, ValueStatus, new_id, utcnow

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
TIME_NAME_HINT = re.compile(r"(time|date|timestamp|datetime|^ts$|^t$)", re.I)
UNIT_IN_HEADER = re.compile(r"^(?P<name>.*?)\s*[\(\[](?P<unit>[^\)\]]{1,16})[\)\]]\s*$")
EPOCH_BASE = pd.Timestamp("1970-01-01", tz="UTC")


class FileValidationError(ValueError):
    """User-facing validation error for an uploaded file."""


@dataclass
class ColumnProfile:
    name: str
    kind: str  # numeric | datetime | boolean | categorical | text | empty
    dtype: str
    missing: int
    non_numeric: int  # non-empty cells that do not parse as numbers
    unique: int
    unit: str = ""
    sample: list[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    duplicate_rows: int
    missing_total: int
    column_profiles: dict[str, ColumnProfile]
    timestamp_candidates: list[str]
    numeric_columns: list[str]


@dataclass
class Dataset:
    id: str
    name: str
    sha256: str
    size_bytes: int
    uploaded_at: datetime
    raw: pd.DataFrame
    profile: DatasetProfile
    timestamp_col: str | None = None
    metric_cols: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    timezone: str = "UTC"
    time_basis: str = "timestamp"  # "timestamp" or "row_index" when the file has no time column
    report: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------- validation
def validate_upload(name: str, data: bytes, max_mb: float = 50.0) -> str:
    """Return the lower-case extension or raise ``FileValidationError``."""
    safe_name = Path(name).name  # never trust directory parts in an upload name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(f"'{safe_name}' is not supported. Upload a .csv or .xlsx file.")
    if not data:
        raise FileValidationError(f"'{safe_name}' is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise FileValidationError(f"'{safe_name}' is {len(data) / 1e6:.1f} MB; the limit is {max_mb:g} MB.")
    if ext == ".xlsx" and not data.startswith(b"PK\x03\x04"):
        raise FileValidationError(f"'{safe_name}' has an .xlsx extension but is not a valid Excel workbook.")
    if ext == ".xls" and not data.startswith(b"\xd0\xcf\x11\xe0"):
        raise FileValidationError(f"'{safe_name}' has an .xls extension but is not a legacy Excel file.")
    if ext == ".csv" and b"\x00" in data[:65536]:
        raise FileValidationError(f"'{safe_name}' contains binary data; it does not look like a text CSV.")
    return ext


def read_table(name: str, data: bytes, max_rows: int = 1_000_000, max_columns: int = 500, sheet: str | int = 0) -> pd.DataFrame:
    ext = Path(name).suffix.lower()
    try:
        if ext == ".csv":
            df = _read_csv(data)
        elif ext == ".xlsx":
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="openpyxl")
        else:
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    except FileValidationError:
        raise
    except ImportError as exc:
        raise FileValidationError(f"Reading {ext} needs an extra package: {exc}. Convert the file to .xlsx or .csv.") from exc
    except Exception as exc:
        raise FileValidationError(f"Could not parse '{name}': {exc}") from exc
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise FileValidationError(f"'{name}' has no data rows.")
    if len(df.columns) > max_columns or len(df) > max_rows:
        raise FileValidationError(f"'{name}' exceeds the limits ({max_columns} columns / {max_rows:,} rows).")
    df.columns = _unique([str(c).strip() or f"column_{i + 1}" for i, c in enumerate(df.columns)])
    return df.reset_index(drop=True)


def list_sheets(data: bytes) -> list[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(data), engine="openpyxl").sheet_names)
    except Exception:
        return []


def _read_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 always decodes
        raise FileValidationError("Could not decode the CSV text.")
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise FileValidationError(f"CSV parse error: {exc}") from exc


def _unique(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        n = seen.get(c, 0)
        out.append(c if n == 0 else f"{c}_{n + 1}")
        seen[c] = n + 1
    return out


# ----------------------------------------------------------------- profiling
def split_unit(header: str) -> tuple[str, str]:
    """'Voltage (V)' -> ('Voltage', 'V'); 'Power [kW]' -> ('Power', 'kW')."""
    m = UNIT_IN_HEADER.match(header.strip())
    if m and m.group("name"):
        return m.group("name").strip(), m.group("unit").strip()
    return header.strip(), ""


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Parse numbers, tolerating thousands separators, %, and unit suffixes like '230 V'."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
    cleaned = (
        s.astype("string")
        .str.strip()
        .str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
        .str.replace(r"\s*(%|[A-Za-zµΩ°/]+)$", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


def parse_timestamps(s: pd.Series, tz: str = "UTC") -> pd.Series:
    """Parse a column to UTC timestamps. Naive values are interpreted in ``tz``.
    Numeric columns are treated as Unix epoch seconds or milliseconds by magnitude."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        med = v.dropna().abs().median() if v.notna().any() else 0
        if 1e11 <= med < 1e14:
            return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
        if 1e8 <= med < 1e11:
            return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC")
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC")
    try:
        return parsed.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    except Exception:
        return parsed.dt.tz_localize("UTC")


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    profiles: dict[str, ColumnProfile] = {}
    ts_candidates: list[tuple[float, str]] = []
    numeric_cols: list[str] = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        present = s.dropna()
        _, unit = split_unit(col)
        num = to_numeric_clean(present) if len(present) else present
        non_numeric = int(num.isna().sum()) if len(present) else 0
        kind = "empty"
        if len(present):
            if pd.api.types.is_bool_dtype(s):
                kind = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                kind = "datetime"
            elif non_numeric / len(present) <= 0.2 and non_numeric < len(present):
                kind = "numeric"
            else:
                kind = "text" if present.astype(str).nunique() > 0.5 * len(present) else "categorical"
        # Timestamp candidates: datetime dtype, name hints, or string columns that parse well.
        score = 0.0
        if len(present):
            if kind == "datetime":
                score = 2.0
            elif kind in ("text", "categorical") or TIME_NAME_HINT.search(col):
                sample = present.head(500)
                try:
                    ok = parse_timestamps(sample).notna().mean()
                except Exception:
                    ok = 0.0
                if ok >= 0.9:
                    score = float(ok) + (0.5 if TIME_NAME_HINT.search(col) else 0.0)
        if score > 0:
            ts_candidates.append((score, col))
            if kind == "numeric" and TIME_NAME_HINT.search(col):
                kind = "datetime"
        if kind == "numeric":
            numeric_cols.append(col)
        profiles[col] = ColumnProfile(
            name=col,
            kind=kind,
            dtype=str(s.dtype),
            missing=missing,
            non_numeric=non_numeric if kind == "numeric" else 0,
            unique=int(present.nunique()) if len(present) else 0,
            unit=unit,
            sample=[_jsonable(v) for v in present.head(3).tolist()],
        )
    ts_candidates.sort(reverse=True)
    return DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        missing_total=int(df.isna().sum().sum()),
        column_profiles=profiles,
        timestamp_candidates=[c for _, c in ts_candidates],
        numeric_columns=numeric_cols,
    )


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v


# ----------------------------------------------------------------- dataset
def load_dataset(name: str, data: bytes, max_mb: float = 50.0, max_rows: int = 1_000_000, max_columns: int = 500, sheet: str | int = 0) -> Dataset:
    validate_upload(name, data, max_mb)
    safe = Path(name).name
    df = read_table(safe, data, max_rows, max_columns, sheet)
    profile = profile_dataframe(df)
    ds = Dataset(
        id=new_id("file"),
        name=safe,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        uploaded_at=utcnow(),
        raw=df,
        profile=profile,
        timestamp_col=profile.timestamp_candidates[0] if profile.timestamp_candidates else None,
        metric_cols=[c for c in profile.numeric_columns if c not in profile.timestamp_candidates[:1]],
    )
    ds.units = {c: profile.column_profiles[c].unit for c in ds.metric_cols}
    return ds


def normalize_dataset(ds: Dataset) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert the selected columns to long-format observations.

    Returns the observations and a data-quality report that lists every row
    that could not be used and why, so nothing is dropped silently.
    """
    df = ds.raw
    report: dict[str, Any] = {
        "rows_in_file": len(df),
        "invalid_timestamps": [],
        "non_numeric_cells": {},
        "missing_cells": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": 0,
        "unsorted_timestamps": False,
        "time_basis": "timestamp",
    }
    if ds.timestamp_col and ds.timestamp_col in df.columns:
        ts = parse_timestamps(df[ds.timestamp_col], ds.timezone)
        bad = ts.isna() & df[ds.timestamp_col].notna()
        report["invalid_timestamps"] = [int(i) for i in df.index[bad | df[ds.timestamp_col].isna()]][:1000]
        valid_ts = ts.dropna()
        report["duplicate_timestamps"] = int(valid_ts.duplicated().sum())
        report["unsorted_timestamps"] = bool(not valid_ts.is_monotonic_increasing)
        ds.time_basis = "timestamp"
    else:
        # No time column: keep order but make it explicit that "time" is row number.
        ts = pd.Series(EPOCH_BASE + pd.to_timedelta(np.arange(len(df)), unit="s"), index=df.index)
        ds.time_basis = "row_index"
        report["time_basis"] = "row_index"
    frames = []
    for col in ds.metric_cols:
        if col not in df.columns:
            continue
        vals = to_numeric_clean(df[col])
        missing = df[col].isna()
        non_num = vals.isna() & ~missing
        report["missing_cells"][col] = int(missing.sum())
        report["non_numeric_cells"][col] = [int(i) for i in df.index[non_num]][:1000]
        keep = vals.notna() & ts.notna()
        if not keep.any():
            continue
        metric, unit_from_header = split_unit(col)
        unit = ds.units.get(col) or unit_from_header
        part = pd.DataFrame(
            {
                "timestamp": ts[keep].values,
                "source_type": SourceType.FILE.value,
                "source_id": ds.id,
                "metric": metric,
                "value": vals[keep].astype(float).values,
                "unit": unit,
                "confidence": 1.0,
                "status": ValueStatus.VERIFIED.value,
                "meta": [
                    {"file": ds.name, "sha256": ds.sha256[:16], "row": int(i), "column": col, "time_basis": ds.time_basis}
                    for i in df.index[keep]
                ],
            }
        )
        frames.append(part)
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OBSERVATION_COLUMNS)
    if not obs.empty:
        obs["timestamp"] = pd.to_datetime(obs["timestamp"], utc=True)
        obs = obs.sort_values(["metric", "timestamp"], kind="stable").reset_index(drop=True)
    report["observations"] = len(obs)
    report["quality_score"] = quality_score(report, len(df), max(1, len(ds.metric_cols)))
    ds.report = report
    return obs, report


def quality_score(report: dict[str, Any], rows: int, n_metrics: int) -> float:
    """0..1 score: fraction of usable cells, penalized for duplicates and missing time."""
    if rows == 0:
        return 0.0
    cells = rows * n_metrics
    bad = sum(report["missing_cells"].values()) + sum(len(v) for v in report["non_numeric_cells"].values())
    bad += len(report["invalid_timestamps"]) * n_metrics
    usable = max(0.0, 1.0 - bad / cells)
    dup_penalty = min(0.2, (report["duplicate_rows"] + report["duplicate_timestamps"]) / rows)
    time_penalty = 0.15 if report["time_basis"] == "row_index" else 0.0
    return round(max(0.0, usable - dup_penalty - time_penalty), 3)
