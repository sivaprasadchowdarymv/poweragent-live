"""SQLite persistence for observations, analyses, alerts, recommendations and
conversations. Screenshots are *not* stored here (see Settings.store_screenshots).

Thread-safe: every call opens a short-lived connection guarded by a lock,
which is simple and safe for the capture/extraction/chat threads.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.stats import epoch_seconds
from core.models import OBSERVATION_COLUMNS, new_id, utcnow

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
    created_at TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, source_type TEXT NOT NULL, source_id TEXT NOT NULL,
    metric TEXT NOT NULL, metric_key TEXT NOT NULL, value REAL NOT NULL,
    unit TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_obs_metric_ts ON observations(metric_key, ts);
CREATE INDEX IF NOT EXISTS ix_obs_source ON observations(source_id);
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL,
    params TEXT NOT NULL, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY, rule_id TEXT NOT NULL, metric TEXT NOT NULL, severity TEXT NOT NULL,
    opened_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, resolved_at TEXT,
    message TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '{}', occurrences INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_alerts_open ON alerts(rule_id, resolved_at);
CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, conversation_id TEXT,
    metric TEXT NOT NULL, source_id TEXT, window TEXT NOT NULL, evidence TEXT NOT NULL,
    text TEXT NOT NULL, criteria TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
    evaluated_at TEXT, outcome TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, title TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    created_at TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, observation_id INTEGER,
    metric TEXT NOT NULL, old_value REAL, new_value REAL NOT NULL, note TEXT NOT NULL DEFAULT ''
);
"""

_SENSITIVE = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[number]"),
    (re.compile(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)\s*[:=]\s*\S+"), r"\1: [redacted]"),
]


def redact(text: str) -> str:
    """Strip obvious secrets before anything from the screen is persisted."""
    for pat, rep in _SENSITIVE:
        text = pat.sub(rep, text)
    return text


def _meta_json(meta: dict[str, Any] | None) -> str:
    """Bounded JSON for message metadata; an oversized result keeps its essentials only."""
    text = json.dumps(meta or {}, default=str)
    if len(text) <= 200_000:
        return text
    slim = {k: v for k, v in (meta or {}).items() if k not in ("result",)}
    res = (meta or {}).get("result") or {}
    slim["result"] = {k: res.get(k) for k in ("kind", "provider", "model", "verification", "warnings", "route", "figure_ids", "report_ids") if k in res}
    return json.dumps(slim, default=str)[:200_000]


def metric_key(name: str) -> str:
    """Case/space/unit-insensitive key used to match the same metric across sources."""
    k = re.sub(r"[\(\[].*?[\)\]]", "", str(name)).lower()
    k = re.sub(r"[^a-z0-9]+", " ", k).strip()
    return k or str(name).lower()


def _ts(dt: Any) -> float:
    return pd.Timestamp(dt).tz_convert("UTC").timestamp() if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt, tz="UTC").timestamp()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mem: sqlite3.Connection | None = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._mem is not None:
                con = self._mem
                try:
                    yield con
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
                return
            con = sqlite3.connect(self.path, timeout=15)
            try:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA synchronous=NORMAL")
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()

    # -------------------------------------------------------------- sources
    def upsert_source(self, source_id: str, type_: str, name: str, meta: dict[str, Any] | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO sources(id,type,name,created_at,meta) VALUES(?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, meta=excluded.meta",
                (source_id, type_, name, utcnow().isoformat(), json.dumps(meta or {}, default=str)),
            )

    def sources(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT s.id, s.type, s.name, s.created_at, s.meta, COUNT(o.id), MIN(o.ts), MAX(o.ts) "
                "FROM sources s LEFT JOIN observations o ON o.source_id = s.id GROUP BY s.id ORDER BY s.created_at"
            ).fetchall()
        return [
            {
                "id": r[0], "type": r[1], "name": r[2], "created_at": r[3], "meta": json.loads(r[4]),
                "observations": r[5],
                "first": datetime.fromtimestamp(r[6], UTC).isoformat() if r[6] else None,
                "last": datetime.fromtimestamp(r[7], UTC).isoformat() if r[7] else None,
            }
            for r in rows
        ]

    def delete_source(self, source_id: str) -> int:
        with self.conn() as c:
            n = c.execute("DELETE FROM observations WHERE source_id=?", (source_id,)).rowcount
            c.execute("DELETE FROM sources WHERE id=?", (source_id,))
        return n

    # -------------------------------------------------------------- observations
    def add_observations(self, df: pd.DataFrame | Iterable[dict[str, Any]], batch: int = 50_000) -> int:
        frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(list(df), columns=OBSERVATION_COLUMNS)
        if frame.empty:
            return 0
        ts = pd.Series(epoch_seconds(pd.to_datetime(frame["timestamp"], utc=True)))
        rows = list(
            zip(
                ts.tolist(),
                frame["source_type"].astype(str),
                frame["source_id"].astype(str),
                frame["metric"].astype(str),
                frame["metric"].astype(str).map(metric_key),
                frame["value"].astype(float),
                frame["unit"].fillna("").astype(str),
                frame["confidence"].astype(float),
                frame["status"].astype(str),
                [json.dumps(m if isinstance(m, dict) else {}, default=str) for m in frame["meta"]],
                strict=True,
            )
        )
        with self.conn() as c:
            for i in range(0, len(rows), batch):
                c.executemany(
                    "INSERT INTO observations(ts,source_type,source_id,metric,metric_key,value,unit,confidence,status,meta) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    rows[i : i + batch],
                )
        return len(rows)

    def query_observations(
        self,
        metric: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        source_id: str | None = None,
        source_type: str | None = None,
        limit: int = 200_000,
    ) -> pd.DataFrame:
        where: list[str] = []
        args: list[Any] = []
        if metric:
            where.append("metric_key=?")
            args.append(metric_key(metric))
        if start is not None:
            where.append("ts>=?")
            args.append(_ts(start))
        if end is not None:
            where.append("ts<=?")
            args.append(_ts(end))
        if source_id:
            where.append("source_id=?")
            args.append(source_id)
        if source_type:
            where.append("source_type=?")
            args.append(source_type)
        sql = "SELECT id, ts, source_type, source_id, metric, value, unit, confidence, status, meta FROM observations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        df = pd.DataFrame(rows, columns=["id", "ts", "source_type", "source_id", "metric", "value", "unit", "confidence", "status", "meta"])
        if df.empty:
            df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
            return df
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df["meta"] = df["meta"].map(json.loads)
        return df.iloc[::-1].reset_index(drop=True)

    def metric_catalog(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT metric_key, MIN(metric), source_id, source_type, status, unit, COUNT(*), MIN(ts), MAX(ts), AVG(confidence) "
                "FROM observations GROUP BY metric_key, source_id, status, unit ORDER BY metric_key"
            ).fetchall()
        return [
            {
                "metric_key": r[0], "metric": r[1], "source_id": r[2], "source_type": r[3], "status": r[4], "unit": r[5],
                "count": r[6], "first": datetime.fromtimestamp(r[7], UTC), "last": datetime.fromtimestamp(r[8], UTC),
                "mean_confidence": round(float(r[9]), 3),
            }
            for r in rows
        ]

    def correct_observation(self, obs_id: int, new_value: float, note: str = "") -> None:
        with self.conn() as c:
            row = c.execute("SELECT metric, value FROM observations WHERE id=?", (obs_id,)).fetchone()
            if row is None:
                raise KeyError(f"Observation {obs_id} not found.")
            c.execute(
                "INSERT INTO corrections(created_at,observation_id,metric,old_value,new_value,note) VALUES(?,?,?,?,?,?)",
                (utcnow().isoformat(), obs_id, row[0], row[1], float(new_value), note),
            )
            c.execute("UPDATE observations SET value=?, status='corrected', confidence=1.0 WHERE id=?", (float(new_value), obs_id))

    def corrections(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT created_at, observation_id, metric, old_value, new_value, note FROM corrections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(["created_at", "observation_id", "metric", "old_value", "new_value", "note"], r, strict=True)) for r in rows]

    def count_observations(self) -> int:
        with self.conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def apply_retention(self, retention_days: int, max_rows: int) -> dict[str, int]:
        cutoff = (utcnow() - timedelta(days=retention_days)).timestamp()
        with self.conn() as c:
            # Retention applies to screen captures (recorded "now"); file data keeps its own historical timestamps.
            old = c.execute("DELETE FROM observations WHERE source_type='screen' AND ts<?", (cutoff,)).rowcount
            total = int(c.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
            over = 0
            if total > max_rows:
                over = c.execute(
                    "DELETE FROM observations WHERE id IN (SELECT id FROM observations WHERE source_type='screen' ORDER BY ts ASC LIMIT ?)",
                    (total - max_rows,),
                ).rowcount
            c.execute("DELETE FROM messages WHERE created_at<?", ((utcnow() - timedelta(days=retention_days)).isoformat(),))
        return {"expired_screen_observations": old, "trimmed_over_cap": over}

    def delete_all_measurements(self) -> int:
        with self.conn() as c:
            n = c.execute("DELETE FROM observations").rowcount
            c.execute("DELETE FROM sources")
            c.execute("DELETE FROM alerts")
            c.execute("DELETE FROM analyses")
            c.execute("DELETE FROM corrections")
        return n

    # -------------------------------------------------------------- analyses
    def save_analysis(self, kind: str, params: dict[str, Any], result: dict[str, Any]) -> str:
        aid = new_id("an")
        with self.conn() as c:
            c.execute(
                "INSERT INTO analyses(id,created_at,kind,params,result) VALUES(?,?,?,?,?)",
                (aid, utcnow().isoformat(), kind, json.dumps(params, default=str), json.dumps(result, default=str)[:200_000]),
            )
        return aid

    def recent_analyses(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT id, created_at, kind, params, result FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "created_at": r[1], "kind": r[2], "params": json.loads(r[3]), "result": json.loads(r[4])} for r in rows]

    # -------------------------------------------------------------- alerts
    def open_alert(self, rule_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            r = c.execute(
                "SELECT id, rule_id, metric, severity, opened_at, last_seen_at, resolved_at, message, evidence, occurrences "
                "FROM alerts WHERE rule_id=? AND resolved_at IS NULL", (rule_id,)
            ).fetchone()
        return self._alert_row(r) if r else None

    def last_resolved(self, rule_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            r = c.execute(
                "SELECT id, rule_id, metric, severity, opened_at, last_seen_at, resolved_at, message, evidence, occurrences "
                "FROM alerts WHERE rule_id=? AND resolved_at IS NOT NULL ORDER BY resolved_at DESC LIMIT 1", (rule_id,)
            ).fetchone()
        return self._alert_row(r) if r else None

    def insert_alert(self, rule_id: str, metric: str, severity: str, message: str, evidence: dict[str, Any], at: datetime) -> str:
        aid = new_id("al")
        with self.conn() as c:
            c.execute(
                "INSERT INTO alerts(id,rule_id,metric,severity,opened_at,last_seen_at,message,evidence) VALUES(?,?,?,?,?,?,?,?)",
                (aid, rule_id, metric, severity, at.isoformat(), at.isoformat(), message, json.dumps(evidence, default=str)),
            )
        return aid

    def touch_alert(self, alert_id: str, at: datetime, evidence: dict[str, Any]) -> None:
        with self.conn() as c:
            c.execute("UPDATE alerts SET last_seen_at=?, occurrences=occurrences+1, evidence=? WHERE id=?", (at.isoformat(), json.dumps(evidence, default=str), alert_id))

    def resolve_alert(self, alert_id: str, at: datetime) -> None:
        with self.conn() as c:
            c.execute("UPDATE alerts SET resolved_at=? WHERE id=?", (at.isoformat(), alert_id))

    def alerts(self, active_only: bool = False, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT id, rule_id, metric, severity, opened_at, last_seen_at, resolved_at, message, evidence, occurrences FROM alerts"
        if active_only:
            sql += " WHERE resolved_at IS NULL"
        sql += " ORDER BY opened_at DESC LIMIT ?"
        with self.conn() as c:
            rows = c.execute(sql, (limit,)).fetchall()
        return [self._alert_row(r) for r in rows]

    @staticmethod
    def _alert_row(r: tuple[Any, ...]) -> dict[str, Any]:
        keys = ["id", "rule_id", "metric", "severity", "opened_at", "last_seen_at", "resolved_at", "message", "evidence", "occurrences"]
        d = dict(zip(keys, r, strict=True))
        d["evidence"] = json.loads(d["evidence"])
        return d

    # -------------------------------------------------------------- recommendations
    def add_recommendation(self, conversation_id: str | None, metric: str, source_id: str | None, window: dict[str, Any], evidence: dict[str, Any], text: str, criteria: dict[str, Any], created_at: datetime | None = None) -> str:
        rid = new_id("rec")
        with self.conn() as c:
            c.execute(
                "INSERT INTO recommendations(id,created_at,conversation_id,metric,source_id,window,evidence,text,criteria) VALUES(?,?,?,?,?,?,?,?,?)",
                (rid, (created_at or utcnow()).isoformat(), conversation_id, metric, source_id, json.dumps(window, default=str), json.dumps(evidence, default=str), text, json.dumps(criteria, default=str)),
            )
        return rid

    def update_recommendation_text(self, rec_id: str, text: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE recommendations SET text=? WHERE id=?", (text[:4000], rec_id))

    def set_recommendation_outcome(self, rec_id: str, status: str, outcome: dict[str, Any]) -> None:
        with self.conn() as c:
            c.execute("UPDATE recommendations SET status=?, evaluated_at=?, outcome=? WHERE id=?", (status, utcnow().isoformat(), json.dumps(outcome, default=str), rec_id))

    def recommendations(self, limit: int = 50, conversation_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, created_at, conversation_id, metric, source_id, window, evidence, text, criteria, status, evaluated_at, outcome FROM recommendations"
        args: list[Any] = []
        if conversation_id:
            sql += " WHERE conversation_id=?"
            args.append(conversation_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        keys = ["id", "created_at", "conversation_id", "metric", "source_id", "window", "evidence", "text", "criteria", "status", "evaluated_at", "outcome"]
        out = []
        for r in rows:
            d = dict(zip(keys, r, strict=True))
            for k in ("window", "evidence", "criteria", "outcome"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    # -------------------------------------------------------------- conversations
    def new_conversation(self, title: str = "") -> str:
        cid = new_id("conv")
        with self.conn() as c:
            c.execute("INSERT INTO conversations(id,created_at,title) VALUES(?,?,?)", (cid, utcnow().isoformat(), title))
        return cid

    def add_message(self, conversation_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO messages(conversation_id,created_at,role,content,meta) VALUES(?,?,?,?,?)",
                (conversation_id, utcnow().isoformat(), role, redact(content)[:20_000], _meta_json(meta)),
            )
            return int(cur.lastrowid or 0)

    def update_message_meta(self, message_id: int, meta: dict[str, Any]) -> None:
        with self.conn() as c:
            c.execute("UPDATE messages SET meta=? WHERE id=?", (_meta_json(meta), int(message_id)))

    def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT id, created_at, role, content, meta FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
        out = []
        for r in rows:
            try:
                meta = json.loads(r[4])
            except ValueError:
                meta = {}
            out.append({"id": r[0], "created_at": r[1], "role": r[2], "content": r[3], "meta": meta})
        return out

    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE conversations SET title=? WHERE id=?", (redact(title)[:80], conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            c.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def conversations(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT c.id, c.created_at, c.title, COUNT(m.id) FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id "
                "GROUP BY c.id ORDER BY c.created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"id": r[0], "created_at": r[1], "title": r[2], "messages": r[3]} for r in rows]

    def delete_conversations(self) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM messages")
            c.execute("DELETE FROM conversations")
