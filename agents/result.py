"""Result objects shared by the pipeline, the chat store and the UI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentStep:
    """One visible step of the agent (stage / action / observation / answer / note).
    Only actions are recorded: the model's hidden reasoning is never requested or shown."""

    kind: str
    text: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    question: str
    kind: str = "answer"            # answer | clarification | offline | error | rate_limited
    answer: str = ""
    route: str = "text"             # text (Groq) | vision (Gemma/Qwen saw the screen) | offline (no LLM)
    sources: List[Dict[str, Any]] = field(default_factory=list)   # cited SourceRef.as_dict()
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # all SourceRef.as_dict()
    warnings: List[str] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    used_llm: bool = False
    llm_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    provider: str = ""
    model: str = ""
    attempts: List[List[str]] = field(default_factory=list)
    images_sent: int = 0
    regenerated: bool = False
    cached: bool = False
    latency: float = 0.0
    stage_latency: Dict[str, float] = field(default_factory=dict)
    figure_ids: List[str] = field(default_factory=list)
    report_ids: List[str] = field(default_factory=list)
    recommendation_id: Optional[str] = None
    data_version: str = ""

    @property
    def verdict(self) -> str:
        return str(self.verification.get("verdict", "n/a"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QueryResult":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        known["steps"] = [AgentStep(**s) if isinstance(s, dict) else s for s in known.get("steps") or []]
        known.setdefault("question", "")
        return cls(**known)
