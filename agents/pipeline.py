"""
The live-data answering pipeline (RAG∞ Ultimate flow, applied to live screens and files).

    question
      │ 1  rate limit + daily budget guard                      (ops.guard / ops.telemetry)
      │ 2  orchestrator: references → intents → tools → route   (agents.orchestrator, no LLM)
      │ 3  answer cache (key includes the DATA VERSION, so a new capture never returns a stale answer)
      │ 4  specialists: the analysis tools run in Python        (agents.tools)
      │ 5  evidence fusion: [REF-n] labels, budget, injection guard (agents.evidence)
      │ 6  route: visual question + screen available → chart crops to the vision model (Gemma / Qwen)
      │                                              otherwise → Groq text model (ReAct tools if needed)
      │ 7  master synthesis                                     (agents.master)
      │ 8  verifier: citations + numeric grounding (+ one bounded regeneration)
      │ 9  recommendation tracking, figures, report, telemetry
      ▼
    QueryResult (answer, verdict, citations, evidence, agent steps, provider/model/tokens/cost)

If no model is reachable, the user still gets the computed facts (clearly labelled as an
automatic summary) - numbers never depend on the LLM.
"""
from __future__ import annotations

import io
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents import fallback
from agents.evidence import EvidenceRegistry, SourceRef
from agents.master import NO_DATA, LiveToolBox, regenerate, run_master
from agents.orchestrator import RoutePlan, route
from agents.references import ConversationState
from agents.result import AgentStep, QueryResult
from agents.tools import ToolCallRecord, ToolContext
from agents.verifier import canonicalize, check, data_age, feedback_text, llm_check
from analysis import recommendations as recs
from config.settings import Settings, get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError
from ops import answer_cache, telemetry
from ops.guard import RateLimiter

log = get_logger("pipeline")
StepFn = Optional[Callable[[AgentStep], None]]
LIMITER = RateLimiter()
_SIDE_EFFECTS = {"generate_trend_plot", "generate_analysis_report", "record_recommendation", "evaluate_recommendation"}


def data_version(runtime: Any) -> str:
    """Changes whenever new measurements, a new capture or a new alert arrive."""
    try:
        n = runtime.db.count_observations()
    except Exception:  # noqa: BLE001
        n = -1
    screen = getattr(runtime.service, "latest_screen", None)
    seq = screen.frame_seq if screen is not None else -1
    try:
        alerts = len(runtime.db.alerts(active_only=True))
    except Exception:  # noqa: BLE001
        alerts = -1
    return f"{n}:{seq}:{alerts}"


def screen_images(runtime: Any, settings: Settings) -> List[Tuple[bytes, str, Dict[str, Any]]]:
    """Chart crops of the LATEST captured frame (or the whole frame when no chart was found),
    downscaled to IMAGE_MAX_PX. Returns [(jpeg bytes, id, chart summary)]."""
    from vision.chart_model import crop_for_chart

    frame = runtime.latest_frame()
    if frame is None:
        return []
    analysis = runtime.service.latest_screen
    charts = list(analysis.charts) if analysis is not None and analysis.frame_seq == frame.seq else []
    # Vision-first: always the whole frame (context: tiles, alarms, legends), then the largest chart crops.
    picks: List[Tuple[Any, str, Dict[str, Any]]] = [(frame.image, "full screen", {"captured_at": frame.captured_at.isoformat()})]
    charts.sort(key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]), reverse=True)
    for ch in charts[: max(0, settings.max_answer_images - 1)]:
        picks.append((crop_for_chart(frame.image, ch), ch.id, ch.summary()))
    out = []
    for i, (img, cid, summary) in enumerate(picks):
        im = img.convert("RGB")
        side = max(settings.image_max_px, 1280) if i == 0 else settings.image_max_px
        if max(im.size) > side:
            im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        out.append((buf.getvalue(), cid, summary))
    return out


def _add_vision_evidence(runtime: Any, registry: EvidenceRegistry, stage: Callable[[str], None]) -> None:
    """The vision model's latest structured reading of the screen and its recent history (inferences over time)."""
    from vision.inference import render_text

    svc = runtime.service
    notes = getattr(svc, "vision_notes", None)
    if not notes:
        return
    age = ""
    try:
        from core.models import utcnow
        import pandas as pd

        age = f", {int((utcnow() - pd.Timestamp(notes['captured_at']).to_pydatetime()).total_seconds())} s ago"
    except Exception:  # noqa: BLE001
        pass
    registry.add_note("VISION-READING", f"{notes.get('model', '')}{age}",
                      f"captured_at: {notes.get('captured_at')}\n" + render_text(notes), data=notes, visual=True)
    hist = list(getattr(svc, "vision_history", []))[-8:-1]
    if hist:
        lines = []
        for h in hist:
            finds = "; ".join(f"[{f['severity']}] {f['finding']}" for f in h.get("findings", [])[:3])
            kp = ", ".join(f"{k['label']}={k['value']:g}" for k in h.get("kpis", [])[:6])
            lines.append(f"{h.get('captured_at', '')[11:19]}: {h.get('summary', '')[:200]} | {kp} | {finds}")
        registry.add_note("VISION-HISTORY", f"{len(hist)} earlier readings", "\n".join(lines), data=hist, visual=True)
    stage(f"Vision readings added: latest + {len(hist)} earlier (inferences over time)")


def _auto_record(ctx: ToolContext, records: List[ToolCallRecord], answer: str) -> Tuple[Optional[str], str]:
    """Tie advice to a metric and a measurable criterion so 'did it work?' can be answered later."""
    metric = ctx.focus.get("metric")
    if not metric:
        return None, ""
    stats = next((r.result.get("statistics") for r in records if r.name == "calculate_statistics" and r.ok), None)
    anomalies = next((r.result for r in records if r.name == "detect_anomalies" and r.ok), None)
    alerts = next((r.result.get("active_alerts") for r in records if r.name == "get_active_alerts" and r.ok), None)
    alert = next((a for a in alerts or [] if str(a.get("metric", "")).lower() == str(metric).lower()), None)
    try:
        crit = recs.suggest_criteria(metric, stats or {}, anomalies, alert)
        win = ctx.focus.get("window") or {}
        rid = ctx.service.db.add_recommendation(ctx.conversation_id, metric, ctx.focus.get("source_id"), win,
                                                {"statistics": stats, "anomaly_count": (anomalies or {}).get("count")},
                                                answer[:1500], crit)
    except (ValueError, KeyError, TypeError) as exc:
        log.info("Could not record recommendation: %s", exc)
        return None, ""
    ctx.focus["last_recommendation_id"] = rid
    return rid, recs.describe_criteria(crit)


def answer_live(question: str, runtime: Any, router: ModelRouter, state: ConversationState,
                settings: Optional[Settings] = None, memory: str = "", previous_question: Optional[str] = None,
                preferences: str = "", on_step: StepFn = None, session_key: str = "local") -> QueryResult:
    settings = settings or runtime.settings
    t_start = time.perf_counter()
    question = (question or "").strip()
    steps: List[AgentStep] = []
    warnings: List[str] = []
    latency: Dict[str, float] = {}

    def emit(step: AgentStep) -> None:
        steps.append(step)
        if on_step:
            try:
                on_step(step)
            except Exception:  # noqa: BLE001
                pass

    def stage(text: str) -> None:
        emit(AgentStep("stage", text))

    # 1. Guards ------------------------------------------------------------------------------
    wait = LIMITER.allow(session_key, settings.rate_limit_per_min, settings.global_rate_limit_per_min)
    if wait:
        return QueryResult(question, kind="rate_limited", route="offline",
                           answer=f"Too many questions in a short time. Please wait {wait:.0f} s and ask again.",
                           latency=round(time.perf_counter() - t_start, 3))

    # 2. Orchestrator ------------------------------------------------------------------------
    ctx = runtime.tool_context(state.focus, state.conversation_id)
    screen_ok = bool(runtime.service.screen_context().get("available"))
    plan: RoutePlan = route(question, state, runtime.service, previous_question, settings.agent_mode,
                            settings.chart_model_enabled, screen_available=screen_ok)
    latency["router"] = plan.latency_ms
    stage(f"Query classified: {', '.join(plan.intents) or 'general'}" + (" (follow-up)" if plan.is_follow_up else ""))
    for note in plan.resolution.notes:
        stage(f"Reference resolved: {note}")
    version = data_version(runtime)
    base = dict(plan=plan.as_dict(), data_version=version)

    if plan.clarification and not plan.calls:
        msg = plan.clarification
        if "No screen" in msg:
            msg += (" Start capture on **Live Monitor** (or press *Capture now*) or upload a CSV/XLSX on **Data**, "
                    "then ask again.")
        return QueryResult(question, kind="clarification", route="offline", answer=msg, steps=steps,
                           latency=round(time.perf_counter() - t_start, 3), **base)

    # 3. Cache ---------------------------------------------------------------------------------
    cache = answer_cache.answers(settings.answer_cache_size, settings.answer_cache_ttl)
    ckey = None
    if not plan.is_follow_up and not (_SIDE_EFFECTS & {n for n, _ in plan.calls}) and "suggest" not in plan.intents:
        ckey = answer_cache.answer_key(question, [version], settings.models_for("master"), settings.models_for("vision"),
                                       settings.agent_mode, settings.image_answers, settings.max_context_chars,
                                       settings.verify_with_llm, preferences)
        hit = cache.get(ckey)
        if hit is not None:
            hit.cached, hit.steps = True, [AgentStep("stage", "Answered from cache (same question, same data, verified earlier)")]
            hit.latency, hit.llm_calls, hit.tokens, hit.cost = round(time.perf_counter() - t_start, 3), 0, 0, 0.0
            if on_step:
                on_step(hit.steps[0])
            return hit

    # 4-5. Specialists → evidence ---------------------------------------------------------------
    t0 = time.perf_counter()
    registry = EvidenceRegistry()
    records: List[ToolCallRecord] = []
    for name, args in plan.calls:
        rec = runtime.registry.call(ctx, name, args)
        records.append(rec)
        registry.add_record(rec)
        emit(AgentStep("action", "", tool=name, args=dict(args)))
        if not rec.ok:
            warnings.append(f"{name}: {rec.result.get('error', 'failed')}")
    latency["specialists"] = (time.perf_counter() - t0) * 1000
    stage(f"Specialists: {len(records)} analysis tool(s) ran in Python ({sum(r.ok for r in records)} ok)")
    if plan.clarification:
        registry.add_note("NOTE", "clarification", plan.clarification)
    _add_vision_evidence(runtime, registry, stage)

    # 6. Route -----------------------------------------------------------------------------------
    images: List[bytes] = []
    image_note = ""
    if plan.visual and settings.image_answers and router.has_role("vision"):
        try:
            picked = screen_images(runtime, settings)
        except Exception as exc:  # noqa: BLE001
            picked = []
            log.warning("Could not prepare screen images: %s", exc)
        for jpeg, cid, summary in picked:
            ref = registry.add_note("SCREEN-IMAGE", cid, f"Image of {cid} from the latest capture (attached).",
                                    data=summary, visual=True)
            images.append(jpeg)
            image_note += ("; " if image_note else "") + f"{cid} {ref.label}"
        if images:
            stage(f"Screen question: {len(images)} image(s) of the latest capture sent to the vision model")
    elif plan.visual:
        why = ("IMAGE_ANSWERS is off (Settings)" if not settings.image_answers else
               "no vision model is configured - add OPENROUTER_API_KEY (or use MODEL_PROFILE=local with Ollama qwen2.5vl)")
        stage(f"Screen question answered from OCR / chart-reader text only: {why}")
        warnings.append(f"The screen image was not shown to a vision model because {why}.")
    if plan.visual and not images and not screen_ok:
        warnings.append("No screen frame has been captured yet: click 'Share a tab' in the sidebar or start capture.")
    stage(f"Route: {'vision model (' + str(len(images)) + ' image(s))' if images else 'text model'}")
    evidence_text = registry.render(budget=settings.max_context_chars)
    stage(f"Evidence fused: {len(registry.refs)} labelled item(s), {len(evidence_text)} characters")

    # 7. Master ------------------------------------------------------------------------------------
    toolbox = LiveToolBox(runtime.registry, ctx, registry, settings)
    outcome = None
    route_name = "vision" if images else "text"
    over_budget = settings.daily_budget_usd > 0 and telemetry.spent_today(settings.data_dir) >= settings.daily_budget_usd
    if over_budget:
        warnings.append(f"Today's AI budget (${settings.daily_budget_usd:.2f}, DAILY_BUDGET_USD) is used up. "
                        "Showing the computed results without an AI answer.")
    elif not router.enabled:
        warnings.append("No AI model is configured (add GROQ_API_KEY, or use MODEL_PROFILE=local with Ollama). "
                        "Showing the computed results without an AI answer.")
    else:
        try:
            try:
                outcome = run_master(question, evidence_text, plan.use_tools, router, settings, toolbox, memory,
                                     preferences, plan.resolution.notes, on_step=emit, images=images,
                                     image_note=image_note)
            except LLMError as exc:
                if not images:
                    raise
                stage(f"Vision model unavailable ({exc.kind}); answering from OCR / chart-reader evidence with the text model")
                tried = ", ".join(f"{m} ({k})" for _, m, k in getattr(exc, "attempts", []) or []) or exc.kind
                warnings.append(f"No vision model could read the screen ({exc.user_message} Tried: {tried}). The answer "
                                "uses the OCR and chart-reader text only. Free Gemma models are often busy - retry in a "
                                "minute, add a few dollars of OpenRouter credit, or switch MODEL_PROFILE.")
                for r in registry.refs:
                    if r.visual:
                        r.visual = False
                images, route_name = [], "text"
                outcome = run_master(question, evidence_text, plan.use_tools, router, settings, toolbox, memory,
                                     preferences, plan.resolution.notes, on_step=emit)
            latency["master"] = outcome.llm_ms
            if outcome.attempts and outcome.attempts[0][2] != "ok":
                failed = ", ".join(f"{m} ({k})" for _, m, k in outcome.attempts if k != "ok")
                stage(f"Model fallback used: {failed} -> {outcome.model}")
        except LLMError as exc:
            warnings.append(f"Showing the computed results without an AI answer. {exc.user_message}")
            outcome = None

    all_records = records + (outcome.records if outcome is not None else [])
    answer = (outcome.answer if outcome is not None else "").strip()
    used_llm = bool(answer)
    if not used_llm:
        route_name = "offline"
        answer = ("**Automatic summary of the computed results** (no AI interpretation):\n\n"
                  + fallback.render(all_records))

    # 8. Verification (+ one bounded regeneration) -------------------------------------------------
    age = data_age(registry.refs)

    def verify(text: str):
        text, cited, bad = canonicalize(text, registry.by_number)
        rep = check(text, registry.refs, question, bad, age)
        if settings.verify_with_llm and used_llm and rep.grounded and not rep.image_values and router.has_role("verifier"):
            rep = llm_check(rep, text, registry.render(budget=12000), router)
        return text, cited, rep

    regenerated = False
    if used_llm:
        answer, cited, report = verify(answer)
        latency["verifier"] = report.latency_ms
        if outcome is not None and report.needs_regeneration and settings.max_regenerations > 0:
            stage("Verification failed: regenerating once with the verifier's feedback")
            try:
                outcome = regenerate(outcome, feedback_text(report), router)
                cand, c_cited, c_rep = verify(outcome.answer.strip())
                latency["verifier"] += c_rep.latency_ms
                latency["master"] = outcome.llm_ms
                answer, cited, report, regenerated = cand, c_cited, c_rep, True
            except LLMError as exc:
                warnings.append(f"Regeneration skipped: {exc.user_message}")
        stage({"verified": "Verification passed", "partial": "Verification passed with notes (see below)"}
              .get(report.verdict, "Verification: issues found (see warnings)"))
        if report.unverified_numbers:
            warnings.append("Unverified numbers: " + ", ".join(report.unverified_numbers)
                            + " - these do not appear in the computed results; treat them with caution.")
        if report.invalid_citations:
            warnings.append(f"The answer cites {', '.join(report.invalid_citations)}, which do not exist.")
        warnings += report.notes
        verification = report.as_dict()
    else:
        cited = []
        verification = {"verdict": "n/a", "notes": ["Computed results shown directly (no AI text to verify)."]}

    # 9. Side effects, metadata ------------------------------------------------------------------------
    rec_id = next((r.result.get("recommendation_id") for r in all_records if r.name == "record_recommendation" and r.ok), None)
    if "suggest" in plan.intents and rec_id is None and used_llm and NO_DATA not in answer.upper()[:40]:
        rec_id, crit = _auto_record(ctx, all_records, answer)
        if rec_id:
            answer += (f"\n\n*Tracking this advice as `{rec_id}` - success criterion: {crit}. Ask \"did your "
                       "suggestion work?\" later to check it against new measurements.*")
    figure_ids = [r.result["figure_id"] for r in all_records if r.name == "generate_trend_plot" and r.ok]
    report_ids = [r.result["report_id"] for r in all_records if r.name == "generate_analysis_report" and r.ok]
    state.update_focus(ctx.focus)
    state.turns += 1
    if registry.injection_lines:
        warnings.append(f"{registry.injection_lines} instruction-like line(s) in the screen/file text were treated "
                        "as data (prompt-injection guard).")

    latency["total"] = (time.perf_counter() - t_start) * 1000
    result = QueryResult(
        question, kind="answer" if used_llm else "offline", answer=answer, route=route_name,
        sources=[r.as_dict() for r in cited], evidence=[r.as_dict() for r in registry.refs],
        warnings=list(dict.fromkeys(w for w in warnings if w)), steps=steps, verification=verification,
        used_llm=used_llm, llm_calls=outcome.llm_calls if outcome else 0, tokens=outcome.tokens if outcome else 0,
        cost=round(outcome.cost if outcome else 0.0, 6), provider=outcome.provider if outcome else "",
        model=outcome.model if outcome else "", attempts=[list(a) for a in (outcome.attempts if outcome else [])],
        images_sent=len(images), regenerated=regenerated, latency=round(latency["total"] / 1000, 2),
        stage_latency={k: round(v, 1) for k, v in latency.items()}, figure_ids=figure_ids, report_ids=report_ids,
        recommendation_id=rec_id, **base)
    try:
        telemetry.record(settings.data_dir, question=question, latency_s=result.latency, tokens=result.tokens,
                         llm_calls=result.llm_calls, provider=result.provider, cached=False, error=not used_llm,
                         kind=route_name, cost_usd=result.cost, model=result.model)
    except Exception:  # noqa: BLE001
        pass
    if ckey and used_llm and result.verdict in ("verified", "partial") and not figure_ids and not report_ids:
        cache.set(ckey, result)
    return result


def sources_from(result: QueryResult) -> List[SourceRef]:
    return [SourceRef.from_dict(d) for d in result.sources]
