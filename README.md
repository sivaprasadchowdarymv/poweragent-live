# ⚡ PowerAgent Live Ultimate

PowerAgent Live Ultimate is a chatbot for live electrical and industrial dashboards. It captures your screen or reads your CSV/XLSX files, computes every number in Python, and has a free LLM explain the results with checked `[REF-n]` citations. It uses the same models and process as **RAG∞ Ultimate**.

```
question → orchestrator (regex intents, references, route; no LLM)
         → specialists (stats / windows / anomalies / alerts / screen / chart model, in Python)
         → evidence registry ([REF-n] labels, budget, prompt-injection guard)
         → master: Groq gpt-oss-120b (ReAct tools only for why/compare/advice)
           or vision: Gemma 4 / Qwen 3.8 with chart crops, for "what's on the screen" questions
         → verifier (citations + numeric grounding, image values capped at "partial", 1 regeneration)
         → cache (keyed by data version) · telemetry · recommendation tracking
```

**Install on Windows:** see [STEP_BY_STEP.md](STEP_BY_STEP.md). The quick version:

```bat
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   (add GROQ_API_KEY and OPENROUTER_API_KEY)
python -m scripts.check_setup
streamlit run app.py
```

**Layout of the code:**

| Folder | Contents |
|---|---|
| `config/` | settings |
| `llm/` | catalog, providers (Groq / OpenRouter / Ollama), router |
| `agents/` | orchestrator, evidence, master, verifier, pipeline, tools |
| `capture/`, `vision/`, `ingest/`, `analysis/`, `storage/` | the live-data engine |
| `ops/` | guard, cache, telemetry |
| `chat/`, `learning/` | conversations, feedback and preferences |
| `ui/` | Streamlit pages |
| `chart_model_server/` | optional Granite Vision / ChartAst server |
| `metrics/` | ChartNet benchmark |
| `tests/` | offline tests |
