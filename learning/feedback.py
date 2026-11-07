"""
Human feedback (👍 / 👎 + reasons), stored persistently as JSON lines.

This is explicit user feedback used for preference adaptation and for
building a future training dataset. No online RLHF happens in the app.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REASONS = ("wrong number", "misread the screen", "missing citation", "wrong metric or window",
           "advice not useful", "too long", "too short", "incomplete", "other")
_lock = threading.Lock()


class FeedbackStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "learning" / "feedback.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, conversation_id: str, message_id: str, question: str, answer: str, rating: int,
               reasons: Optional[List[str]] = None, comment: str = "", provider: str = "", model: str = "",
               metrics: Optional[Dict[str, Any]] = None, regenerated_from: str = "") -> Dict[str, Any]:
        entry = {"ts": time.time(), "conversation_id": conversation_id, "message_id": message_id,
                 "question": question, "answer": answer, "rating": 1 if rating > 0 else -1,
                 "reasons": [r for r in (reasons or []) if r in REASONS], "comment": comment[:1000],
                 "provider": provider, "model": model, "metrics": metrics or {},
                 "regenerated_from": regenerated_from}
        with _lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def stats(self) -> Dict[str, Any]:
        entries = self.all()
        up = sum(1 for e in entries if e["rating"] > 0)
        down = len(entries) - up
        reasons: Dict[str, int] = {}
        for e in entries:
            for r in e.get("reasons", []):
                reasons[r] = reasons.get(r, 0) + 1
        return {"total": len(entries), "up": up, "down": down,
                "ratio": round(up / len(entries), 3) if entries else None, "reasons": reasons}
