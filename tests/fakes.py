"""Scripted stand-in for llm.model_router.ModelRouter (offline tests, no keys)."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from llm.model_router import RoutedResult
from llm.providers import ChatResult, LLMError, ToolCall


def cite_first_number(messages: List[Dict[str, Any]]) -> str:
    """Answer with a number copied from the first evidence item, cited (a well-behaved model)."""
    user = next(m["content"] for m in messages if m["role"] == "user")
    m = re.search(r"\[REF-(\d+)[^\]]*\]\n(.*?)(?:\n\n|$)", user, re.S)
    if not m:
        return "NO DATA: nothing to analyse."
    nums = re.findall(r"=(-?\d+(?:\.\d+)?)", m.group(2))
    num = next((n for n in nums if abs(float(n)) > 3), nums[0] if nums else "")
    return f"The evidence shows a value of {num} [REF-{m.group(1)}]." if num else f"See [REF-{m.group(1)}]."


class FakeRouter:
    def __init__(self, script: Optional[Callable[[str, List[Dict[str, Any]], Dict[str, Any]], Any]] = None,
                 roles=("master", "fast", "verifier", "vision"), fail_vision: bool = False):
        self.calls: List[Dict[str, Any]] = []
        self.script = script
        self.roles = set(roles)
        self.fail_vision = fail_vision
        self.enabled = True

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def chat(self, role, messages, tools=None, tool_choice="auto", images=None, **kw) -> RoutedResult:
        self.calls.append({"role": role, "messages": messages, "tools": tools, "images": list(images or []),
                           "provider": "openrouter" if role == "vision" else "groq"})
        if role == "vision" and self.fail_vision:
            raise LLMError("vision down", "unavailable", "openrouter")
        out = self.script(role, messages, {"tools": tools, "tool_choice": tool_choice, "images": images}) if self.script \
            else cite_first_number(messages)
        if isinstance(out, ChatResult):
            res = out
        elif isinstance(out, tuple):  # ("tool", name, args)
            res = ChatResult("", [ToolCall("c1", out[1], json.dumps(out[2]))], tokens=10, provider="groq", model="fake")
        else:
            res = ChatResult(str(out), tokens=20, provider="openrouter" if role == "vision" else "groq", model=f"fake-{role}")
        return RoutedResult(res, [(res.provider, res.model, "ok")], 1.0)
