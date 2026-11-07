"""Shared data types. Every observation carries its provenance."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd


class SourceType(str, Enum):
    FILE = "file"
    SCREEN = "screen"
    MANUAL = "manual"


class ValueStatus(str, Enum):
    VERIFIED = "verified"  # exact structured data (file cells)
    ESTIMATED = "estimated"  # pixel-derived or OCR-derived from a screenshot
    CORRECTED = "corrected"  # a screen estimate that the user corrected by hand


OBSERVATION_COLUMNS = [
    "timestamp",
    "source_type",
    "source_id",
    "metric",
    "value",
    "unit",
    "confidence",
    "status",
    "meta",
]


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class Observation:
    timestamp: datetime
    source_type: SourceType
    source_id: str
    metric: str
    value: float
    unit: str = ""
    confidence: float = 1.0
    status: ValueStatus = ValueStatus.VERIFIED
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_type"] = self.source_type.value
        row["status"] = self.status.value
        return row


def observations_frame(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows or [], columns=OBSERVATION_COLUMNS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


@dataclass
class TimeWindow:
    start: datetime
    end: datetime
    label: str = ""
    anchor: str = "now"  # "now" or "latest-data" — states what the window is relative to

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    def describe(self) -> str:
        fmt = "%Y-%m-%d %H:%M:%S"
        return f"{self.start.strftime(fmt)} → {self.end.strftime(fmt)} UTC" + (f" ({self.label})" if self.label else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "label": self.label,
            "anchor": self.anchor,
        }
