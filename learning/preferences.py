"""
User preference profile, adapted from explicit feedback (same design as RAG∞ Ultimate).

This is PREFERENCE ADAPTATION, not RLHF: simple, transparent rules update a small
profile that is injected into the master agent's prompt. Every automatic change is
logged so the user can see and undo it.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

LENGTHS = ("short", "medium", "long")
DEPTHS = ("basic", "medium", "high")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


@dataclass
class Preferences:
    response_length: str = "medium"
    technical_depth: str = "high"
    tables: bool = True          # compare several values in a Markdown table
    sections: bool = True        # Facts / Estimates / Possible explanations / Recommendations
    history: List[Dict] = field(default_factory=list)  # automatic changes, newest last

    def to_prompt(self) -> str:
        length = {"short": "Keep answers brief: 2-4 sentences.",
                  "medium": "Use a moderate length.",
                  "long": "Give thorough answers with detailed explanations."}[self.response_length]
        depth = {"basic": "Explain in simple terms for a non-specialist.",
                 "medium": "Assume general engineering knowledge.",
                 "high": "Use precise, expert-level engineering language."}[self.technical_depth]
        extra = []
        if self.tables:
            extra.append("Present comparisons of several values as a Markdown table.")
        if self.sections:
            extra.append("When useful, separate Facts, Estimates, Possible explanations and Recommendations.")
        return " ".join([length, depth, *extra])


class PreferenceStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "learning" / "preferences.json"

    def load(self) -> Preferences:
        raw = read_json(self.path, {}) or {}
        return Preferences(**{k: v for k, v in raw.items() if k in Preferences.__dataclass_fields__})

    def save(self, prefs: Preferences) -> None:
        write_json_atomic(self.path, asdict(prefs))

    def apply_feedback(self, rating: int, reasons: List[str]) -> List[str]:
        """Update the profile from one piece of feedback; returns human-readable changes."""
        prefs, changes = self.load(), []

        def shift(attr: str, scale, step: int, why: str) -> None:
            cur = scale.index(getattr(prefs, attr))
            new = scale[max(0, min(len(scale) - 1, cur + step))]
            if new != getattr(prefs, attr):
                changes.append(f"{attr.replace('_', ' ')}: {getattr(prefs, attr)} → {new} ({why})")
                setattr(prefs, attr, new)

        if rating < 0:
            if "too long" in reasons:
                shift("response_length", LENGTHS, -1, "answer was too long")
            if "too short" in reasons or "incomplete" in reasons:
                shift("response_length", LENGTHS, +1, "answer was too short or incomplete")
        if changes:
            prefs.history = (prefs.history + [{"ts": time.time(), "changes": changes}])[-50:]
            self.save(prefs)
        return changes
