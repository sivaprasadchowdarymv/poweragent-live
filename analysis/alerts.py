"""Configurable alert rules with persistence, cooldown, deduplication and
resolution tracking. Alerts only notify — they never trigger external actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from core.models import new_id, utcnow
from storage.db import Database, metric_key
from analysis.stats import InsufficientData, detect_anomalies, prepare_series

SEVERITIES = ("info", "warning", "critical")


@dataclass
class AlertRule:
    metric: str
    kind: str  # above | below | outside | anomaly | stale
    threshold: float | None = None
    lower: float | None = None
    persistence_s: float = 0.0  # condition must hold this long before opening
    severity: str = "warning"
    cooldown_s: float = 120.0  # after resolution, do not reopen within this time
    stale_after_s: float = 30.0
    enabled: bool = True
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id("rule")
        if self.kind not in ("above", "below", "outside", "anomaly", "stale"):
            raise ValueError(f"Unknown alert kind '{self.kind}'.")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Severity must be one of {SEVERITIES}.")
        if self.kind in ("above", "below") and self.threshold is None:
            raise ValueError("Threshold rules need a threshold value.")
        if self.kind == "outside" and (self.threshold is None or self.lower is None):
            raise ValueError("'outside' rules need lower and upper limits.")
        if self.persistence_s < 0 or self.cooldown_s < 0:
            raise ValueError("Persistence and cooldown must be non-negative.")

    def describe(self) -> str:
        if self.kind == "above":
            cond = f"{self.metric} > {self.threshold:g}"
        elif self.kind == "below":
            cond = f"{self.metric} < {self.threshold:g}"
        elif self.kind == "outside":
            cond = f"{self.metric} outside [{self.lower:g}, {self.threshold:g}]"
        elif self.kind == "anomaly":
            cond = f"anomaly in {self.metric}"
        else:
            cond = f"no new {self.metric} data for {self.stale_after_s:g}s"
        return cond + (f" for ≥{self.persistence_s:g}s" if self.persistence_s else "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertEngine:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.rules: dict[str, AlertRule] = {}

    def add_rule(self, rule: AlertRule) -> AlertRule:
        self.rules[rule.id] = rule
        return rule

    def remove_rule(self, rule_id: str) -> None:
        self.rules.pop(rule_id, None)

    def evaluate(self, now: datetime | None = None, lookback_s: float = 900) -> list[dict[str, Any]]:
        """Evaluate every enabled rule against recent stored observations.
        Returns a list of events: opened / updated / resolved / suppressed."""
        now = now or utcnow()
        events: list[dict[str, Any]] = []
        for rule in list(self.rules.values()):
            if not rule.enabled:
                continue
            start = now - timedelta(seconds=max(lookback_s, rule.persistence_s * 2 + 60))
            df = self.db.query_observations(metric=rule.metric, start=start, end=now)
            events.extend(self._evaluate_rule(rule, df, now))
        return events

    def _evaluate_rule(self, rule: AlertRule, df: pd.DataFrame, now: datetime) -> list[dict[str, Any]]:
        active, evidence = self._condition(rule, df, now)
        current = self.db.open_alert(rule.id)
        if active:
            if current:
                self.db.touch_alert(current["id"], now, evidence)
                return [{"event": "updated", "alert_id": current["id"], "rule": rule.describe()}]
            last = self.db.last_resolved(rule.id)
            if last and last["resolved_at"] and (now - datetime.fromisoformat(last["resolved_at"])).total_seconds() < rule.cooldown_s:
                return [{"event": "suppressed", "rule": rule.describe(), "reason": "cooldown"}]
            aid = self.db.insert_alert(rule.id, rule.metric, rule.severity, rule.describe(), evidence, now)
            return [{"event": "opened", "alert_id": aid, "rule": rule.describe(), "severity": rule.severity}]
        if current:
            self.db.resolve_alert(current["id"], now)
            return [{"event": "resolved", "alert_id": current["id"], "rule": rule.describe()}]
        return []

    def _condition(self, rule: AlertRule, df: pd.DataFrame, now: datetime) -> tuple[bool, dict[str, Any]]:
        if rule.kind == "stale":
            last = df["timestamp"].max() if not df.empty else None
            age = (pd.Timestamp(now) - last).total_seconds() if last is not None else float("inf")
            return age > rule.stale_after_s, {"last_observation": last.isoformat() if last is not None else None, "age_s": age}
        if df.empty:
            return False, {}
        df = df.sort_values("timestamp")
        latest = df.iloc[-1]
        if rule.kind == "anomaly":
            try:
                sd = prepare_series(df, rule.metric, str(latest["unit"]))
                res = detect_anomalies(sd)
            except InsufficientData:
                return False, {}
            recent = [a for a in res["anomalies"] if pd.Timestamp(a["timestamp"]) >= pd.Timestamp(latest["timestamp"]) - pd.Timedelta(seconds=max(rule.persistence_s, 10))]
            return bool(recent), {"anomalies": recent[-5:], "method": res["method"]}

        def violates(v: float) -> bool:
            if rule.kind == "above":
                return v > float(rule.threshold)  # type: ignore[arg-type]
            if rule.kind == "below":
                return v < float(rule.threshold)  # type: ignore[arg-type]
            return v > float(rule.threshold) or v < float(rule.lower)  # type: ignore[arg-type]

        if not violates(float(latest["value"])):
            return False, {"latest": float(latest["value"]), "at": latest["timestamp"].isoformat()}
        # Persistence: the condition must have held continuously for persistence_s.
        flags = df["value"].map(violates)
        since = latest["timestamp"]
        for ts, flag in zip(df["timestamp"][::-1], flags[::-1], strict=True):
            if not flag:
                break
            since = ts
        held = (latest["timestamp"] - since).total_seconds()
        ok = held >= rule.persistence_s
        return ok, {
            "latest": float(latest["value"]),
            "unit": latest["unit"],
            "at": latest["timestamp"].isoformat(),
            "violating_since": since.isoformat(),
            "held_s": held,
            "source_id": latest["source_id"],
            "status": latest["status"],
        }

    def save(self, path: Any) -> None:
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([r.to_dict() for r in self.rules.values()], indent=2), encoding="utf-8")

    def load(self, path: Any) -> int:
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return 0
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return 0
        n = 0
        for it in items if isinstance(items, list) else []:
            try:
                self.add_rule(AlertRule(**{k: v for k, v in it.items() if k in AlertRule.__dataclass_fields__}))
                n += 1
            except (TypeError, ValueError):
                continue
        return n

    def rules_for(self, metric: str) -> list[AlertRule]:
        k = metric_key(metric)
        return [r for r in self.rules.values() if metric_key(r.metric) == k]
