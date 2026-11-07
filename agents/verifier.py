"""
Verification Agent: check the master's answer before the user sees it (RAG∞ Ultimate design,
adapted to live numeric data).

Deterministic checks (always on, zero tokens, ~1 ms):
  citation precision   cited [REF-n] labels that point to real evidence
  citation coverage    factual paragraphs that carry at least one citation
  numerical grounding  every number in the answer equals (after the rounding it is displayed
                       with) a number in the evidence it cites, or in the question
  image-read values    numbers in sentences citing a SCREEN-IMAGE (a chart crop the vision
                       model looked at) cannot be checked against computed data: they are
                       accepted, but the verdict is capped at "partial"
  causal wording       "caused by" without hedging, when the evidence is correlational
  stale data           calling data "current" when the newest point is minutes old

Optional LLM cross-check (VERIFY_WITH_LLM=true, off by default to save tokens): the
"verifier" role (Groq Llama 3.3 70B, a different family from the master) checks the
answer against the same evidence. It can only make the verdict stricter.

These are application-level checks, not proofs of correctness.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

from agents.evidence import SourceRef
from config.settings import get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError

log = get_logger("verifier")

CITE = re.compile(r"\[REF-(\d+)[^\]]*\]")
NUM_RE = re.compile(r"(?<![\w.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
TIME_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]?[\d:.+Z-]*|\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?")
_IDS = re.compile(r"\b(?:chart|fig|rec|rep|conv|ds|src|scr)[_-]?[0-9a-f]+\b|\bREF-\d+\b", re.I)
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|#+)\s")
CAUSAL = re.compile(r"\b(caused|causes|is causing|because of|due to|leads to|led to|resulted in|drives|driven by)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|possibl\w*|likely|suggests?|hypothes\w*|associated|coincid\w*|not (necessarily|proof)|correlat\w*)\b", re.I)
CURRENT = re.compile(r"\b(currently|right now|at the moment|is now|live value)\b", re.I)
NO_DATA = "NO DATA"


@dataclass
class VerificationReport:
    grounded: bool = True
    verdict: str = "verified"  # verified | partial | unverified | n/a
    citation_precision: float = 1.0
    citation_coverage: float = 1.0
    citation_accuracy: float = 1.0
    numerical_score: float = 1.0
    completeness: float = 1.0
    checked_numbers: int = 0
    image_values: int = 0
    unsupported_claims: List[str] = field(default_factory=list)
    unverified_numbers: List[str] = field(default_factory=list)
    miscited: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    causal_without_hedge: bool = False
    stale_called_current: bool = False
    needs_regeneration: bool = False
    llm_checked: bool = False
    llm_model: str = ""
    llm_cost: float = 0.0
    notes: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ numbers
def collect_numbers(obj: Any, out: Optional[Set[float]] = None) -> Set[float]:
    """Every finite number inside a result (dict / list / str), timestamps excluded."""
    out = out if out is not None else set()
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.add(float(obj))
    elif isinstance(obj, str):
        for m in NUM_RE.findall(TIME_LIKE.sub(" ", obj)):
            try:
                out.add(float(m.replace(",", "").replace("−", "-")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, Iterable):
        for v in obj:
            collect_numbers(v, out)
    return out


def _decimals(s: str) -> int:
    s = s.split("e")[0].split("E")[0]
    return len(s.split(".")[1]) if "." in s else 0


def _sigfigs(token: str) -> int:
    digits = token.lstrip("+-−").split("e")[0].split("E")[0].replace(".", "").lstrip("0")
    return max(1, len(digits))


def matches(token: str, value: float, evidence: Iterable[float]) -> bool:
    """True if `value` equals an evidence value after the rounding the answer displays
    (decimal places or significant figures), also allowing a fraction shown as a percentage.
    No looser tolerance is accepted."""
    tol_display = 0.5 * 10 ** (-_decimals(token)) + 1e-9
    sig = _sigfigs(token)
    for e in evidence:
        for cand in (e, e * 100):
            a, c = abs(value), abs(cand)
            if abs(a - c) <= tol_display:
                return True
            if c:
                rounded = float(f"{c:.{sig}g}")
                if abs(rounded - a) <= 1e-9 * max(1.0, a):
                    return True
    return False


def answer_numbers(text: str) -> List[Tuple[str, float]]:
    """(token, value) for every number that needs grounding; citations, ids, timestamps,
    list markers, small counts and years are skipped."""
    clean = _IDS.sub(" ", CITE.sub(" ", TIME_LIKE.sub(" ", _LIST_MARKER.sub(" ", text))))
    out = []
    for tok in NUM_RE.findall(clean):
        raw = tok.replace(",", "").replace("−", "-")
        try:
            val = float(raw)
        except ValueError:
            continue
        if val == int(val) and (abs(val) <= 3 or 1900 <= val <= 2100):
            continue
        out.append((raw, val))
    return out


def _paragraphs(answer: str) -> List[str]:
    parts = re.split(r"\n\s*\n|\n(?=\s*[-*•|]\s)|\n(?=\s*\d+\.\s)", answer)
    return [p.strip() for p in parts if p.strip()]


def _sentences(par: str) -> List[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z*$])|\n", par) if x.strip()]


def _is_factual(par: str) -> bool:
    plain = re.sub(r"\*\*[^*]+:?\*\*:?", "", par).strip(" #|-")
    if not plain or NO_DATA in plain.upper():
        return False
    return bool(answer_numbers(plain))


# ------------------------------------------------------------------ check
def check(answer: str, refs: Sequence[SourceRef], question: str, invalid_citations: Sequence[str] = (),
          data_age_s: Optional[float] = None) -> VerificationReport:
    t0 = time.perf_counter()
    rep = VerificationReport(invalid_citations=list(invalid_citations))
    if not answer.strip():
        rep.verdict, rep.completeness, rep.grounded = "n/a", 0.0, False
        return rep
    by_num = {r.ref_num: r for r in refs}
    ref_nums = {r.ref_num: collect_numbers(r.data if r.data is not None else r.text) for r in refs}
    visual = {r.ref_num for r in refs if r.visual}
    all_nums: Set[float] = set().union(*ref_nums.values()) if ref_nums else set()
    free = collect_numbers(question)

    labels = CITE.findall(answer)
    total = len(labels) + len(invalid_citations)
    rep.citation_precision = round(len(labels) / total, 3) if total else 0.0
    factual = [p for p in _paragraphs(answer) if _is_factual(p)]
    if factual:
        rep.citation_coverage = round(sum(1 for p in factual if CITE.search(p)) / len(factual), 3)

    n_total = n_ok = cited_total = cited_ok = 0
    for par in _paragraphs(answer):
        par_refs = [int(x) for x in CITE.findall(par)]
        for sent in _sentences(par):
            cited = [int(x) for x in CITE.findall(sent)] or par_refs
            cited = [r for r in cited if r in by_num]
            own = [v for r in cited for v in ref_nums.get(r, ())]
            for tok, val in answer_numbers(sent):
                n_total += 1
                if cited:
                    cited_total += 1
                if cited and matches(tok, val, own):
                    n_ok += 1
                    cited_ok += 1
                elif visual and (visual & set(cited) or not cited):
                    # the vision model looked at the screen: a value it read there cannot be checked against
                    # computed data - accepted as a visual estimate, verdict capped at "partial"
                    rep.image_values += 1
                    n_ok += 1
                    cited_ok += 1
                elif matches(tok, val, free):
                    n_ok += 1
                    cited_ok += 1 if cited else 0
                elif matches(tok, val, all_nums):
                    n_ok += 1
                    if cited:
                        holder = next((r for r, vals in ref_nums.items() if matches(tok, val, vals)), None)
                        rep.miscited.append(f"{tok} is in REF-{holder}, not REF-{', REF-'.join(map(str, cited))}")
                else:
                    rep.unverified_numbers.append(tok)
                    rep.unsupported_claims.append(CITE.sub("", sent).strip()[:160])
    rep.checked_numbers = n_total
    if n_total:
        rep.numerical_score = round(n_ok / n_total, 3)
    if cited_total:
        rep.citation_accuracy = round(cited_ok / cited_total, 3)
    rep.unverified_numbers = list(dict.fromkeys(rep.unverified_numbers))[:8]
    rep.unsupported_claims = list(dict.fromkeys(rep.unsupported_claims))[:5]
    rep.miscited = list(dict.fromkeys(rep.miscited))[:5]

    if CAUSAL.search(answer) and not HEDGE.search(answer):
        rep.causal_without_hedge = True
        rep.notes.append("The evidence shows association only; no causal link has been established from this data.")
    if data_age_s is not None and data_age_s > 120 and CURRENT.search(answer):
        rep.stale_called_current = True
        rep.notes.append(f"The latest data point is {data_age_s:.0f} s old, so it may not reflect the current state.")
    if len(answer.strip()) < 15 or answer.rstrip().endswith((",", " and", " the", "(")):
        rep.completeness = 0.5

    no_data = NO_DATA in answer.upper()[:80]
    rep.grounded = (not invalid_citations and not rep.unverified_numbers
                    and (rep.citation_precision > 0 or n_total == 0 or no_data))
    rep.needs_regeneration = not rep.grounded
    if rep.image_values:
        rep.notes.append(f"{rep.image_values} value(s) were read from the screen image by the vision model; "
                         "they are visual estimates - check them on the screen.")
    rep.verdict = ("unverified" if not rep.grounded else
                   "partial" if (rep.image_values or rep.miscited or rep.citation_coverage < 0.75
                                 or rep.causal_without_hedge or rep.stale_called_current or rep.completeness < 1)
                   else "verified")
    rep.latency_ms = (time.perf_counter() - t0) * 1000
    return rep


def llm_check(rep: VerificationReport, answer: str, evidence_text: str, router: ModelRouter) -> VerificationReport:
    """Second opinion from the verifier model. It can only make the verdict stricter."""
    t0 = time.perf_counter()
    prompt = ("You are a strict fact-checker for answers about live measurement data. Compare the ANSWER with the "
              "EVIDENCE. A claim is unsupported if its value, unit, time window or meaning is not stated in the "
              "evidence. Ignore style and formatting. "
              'Reply with JSON only: {"grounded": true|false, "unsupported_claims": ["short quote", ...]}.\n\n'
              f"EVIDENCE:\n{evidence_text[:12000]}\n\nANSWER:\n{answer[:4000]}")
    try:
        routed = router.chat("verifier", [{"role": "user", "content": prompt}], max_tokens=600, json_mode=True,
                             temperature=0.0)
        data = json.loads(re.search(r"\{.*\}", routed.result.content, re.S).group(0))  # type: ignore[union-attr]
        rep.llm_checked, rep.llm_model, rep.llm_cost = True, routed.result.model, routed.result.cost
        claims = [str(c)[:160] for c in (data.get("unsupported_claims") or []) if str(c).strip()][:5]
        if data.get("grounded") is False:
            rep.grounded, rep.needs_regeneration, rep.verdict = False, True, "unverified"
            rep.unsupported_claims += claims
            rep.notes.append(f"The verifier model ({routed.result.model}) flagged unsupported statements.")
        elif claims and rep.verdict == "verified":
            rep.verdict = "partial"
            rep.notes.append("The verifier model raised minor doubts: " + " | ".join(claims[:2]))
    except (LLMError, AttributeError, ValueError, TypeError) as exc:
        rep.notes.append(f"LLM verification skipped ({type(exc).__name__}).")
    rep.latency_ms += (time.perf_counter() - t0) * 1000
    return rep


def feedback_text(rep: VerificationReport) -> str:
    parts = []
    if rep.invalid_citations:
        parts.append(f"These citations do not exist: {', '.join(rep.invalid_citations)}.")
    if rep.unverified_numbers:
        parts.append("These numbers do not appear in the evidence: " + ", ".join(rep.unverified_numbers)
                     + ". Do not compute new numbers; use only values from the evidence (or call a tool).")
    if rep.miscited:
        parts.append("Fix these citations: " + "; ".join(rep.miscited))
    if rep.citation_precision == 0 and rep.checked_numbers:
        parts.append("Cite the evidence label, e.g. [REF-1], after every number.")
    return " ".join(parts) or "Make sure every number is taken from the evidence and cited."


def canonicalize(answer: str, by_number: Callable[[int], Optional[SourceRef]]) -> Tuple[str, List[SourceRef], List[str]]:
    """Rewrite citations to their true labels. Returns (answer, cited refs, invalid labels)."""
    cited: List[SourceRef] = []
    bad: List[str] = []

    def _sub(m: "re.Match[str]") -> str:
        ref = by_number(int(m.group(1)))
        if ref is None:
            bad.append(m.group(0))
            return m.group(0)
        if ref not in cited:
            cited.append(ref)
        return ref.label

    return CITE.sub(_sub, answer), cited, list(dict.fromkeys(bad))


def data_age(refs: Sequence[SourceRef]) -> Optional[float]:
    ages = []
    for r in refs:
        if isinstance(r.data, dict):
            for key in ("data_age_seconds", "age_seconds"):
                v = r.data.get(key)
                if isinstance(v, (int, float)):
                    ages.append(float(v))
    return min(ages) if ages else None
