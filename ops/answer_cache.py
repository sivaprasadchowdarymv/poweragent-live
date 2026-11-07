"""TTL + LRU cache for verified answers and query embeddings.

Only standalone questions (no conversation memory) with an answer that passed
verification are cached, so a cache hit can never return an unchecked answer.
"""
from __future__ import annotations

import copy
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Sequence


class TTLCache:
    def __init__(self, maxsize: int, ttl: float) -> None:
        self.maxsize, self.ttl = maxsize, ttl
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if self.maxsize <= 0 or self.ttl <= 0:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None or time.time() - item[0] > self.ttl:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(item[1])

    def set(self, key: str, value: Any) -> None:
        if self.maxsize <= 0 or self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.time(), copy.deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def normalize_question(q: str) -> str:
    return " ".join((q or "").lower().split()).rstrip(" ?.!")


def answer_key(question: str, doc_ids: Sequence[str], *parts: Any) -> str:
    raw = "\x1f".join([normalize_question(question), ",".join(sorted(doc_ids))] + [repr(p) for p in parts])
    return hashlib.sha256(raw.encode()).hexdigest()


_ANSWERS: Optional[TTLCache] = None


def answers(maxsize: int, ttl: float) -> TTLCache:
    global _ANSWERS
    if _ANSWERS is None or (_ANSWERS.maxsize, _ANSWERS.ttl) != (maxsize, ttl):
        _ANSWERS = TTLCache(maxsize, ttl)
    return _ANSWERS
