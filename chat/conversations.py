"""
Persistent multi-conversation chat with controlled memory (RAG∞ Ultimate design),
stored in the app's SQLite database so chat history follows the same retention
and "delete everything" controls as the measurements.

Only the last MEMORY_TURNS exchanges (questions + shortened answers, citations
removed) are given to the model: memory stays bounded however long the chat gets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from storage.db import Database

_CITE = re.compile(r"\[REF-\d+[^\]]*\]")


@dataclass
class ChatMessage:
    id: int
    role: str  # user | assistant
    content: str
    created: str
    meta: Dict[str, Any] = field(default_factory=dict)  # assistant: {"result": QueryResult dict, "feedback": {...}}

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self.meta.get("result")


@dataclass
class Conversation:
    id: str
    title: str
    created: str
    messages: List[ChatMessage] = field(default_factory=list)

    def last_question(self) -> Optional[str]:
        return next((m.content for m in reversed(self.messages) if m.role == "user"), None)

    def memory(self, turns: int, chars: int = 500) -> str:
        """Last `turns` Q&A pairs, answers shortened and citations stripped."""
        pairs, q = [], None
        for m in self.messages:
            if m.role == "user":
                q = m.content
            elif q is not None:
                pairs.append((q, _CITE.sub("", m.content).strip()[:chars]))
                q = None
        return "\n".join(f"Q: {a}\nA: {b}" for a, b in pairs[-turns:]) if turns > 0 else ""


class ConversationStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, title: str = "New chat") -> Conversation:
        cid = self.db.new_conversation(title)
        return Conversation(cid, title, "")

    def get(self, conversation_id: str) -> Optional[Conversation]:
        row = next((c for c in self.db.conversations(limit=10_000) if c["id"] == conversation_id), None)
        if row is None:
            return None
        msgs = [ChatMessage(m["id"], m["role"], m["content"], m["created_at"], m["meta"] or {})
                for m in self.db.messages(conversation_id)]
        return Conversation(row["id"], row["title"] or "New chat", row["created_at"], msgs)

    def list(self, limit: int = 30) -> List[Dict[str, Any]]:
        return self.db.conversations(limit=limit)

    def add_user(self, conv: Conversation, text: str) -> ChatMessage:
        mid = self.db.add_message(conv.id, "user", text)
        msg = ChatMessage(mid, "user", text, "")
        conv.messages.append(msg)
        if conv.title in ("", "New chat"):
            conv.title = text.strip().replace("\n", " ")[:60] or "New chat"
            self.db.set_conversation_title(conv.id, conv.title)
        return msg

    def add_assistant(self, conv: Conversation, text: str, result: Dict[str, Any]) -> ChatMessage:
        meta = {"result": result}
        mid = self.db.add_message(conv.id, "assistant", text, meta)
        msg = ChatMessage(mid, "assistant", text, "", meta)
        conv.messages.append(msg)
        return msg

    def set_feedback(self, msg: ChatMessage, feedback: Dict[str, Any]) -> None:
        msg.meta["feedback"] = feedback
        self.db.update_message_meta(msg.id, msg.meta)

    def delete(self, conversation_id: str) -> None:
        self.db.delete_conversation(conversation_id)
