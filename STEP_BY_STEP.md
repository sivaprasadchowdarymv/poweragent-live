# PowerAgent Live Ultimate — Windows installation, step by step

Everything here is free: Groq free tier (text), OpenRouter free Gemma 4 (screen images), Tesseract OCR, and optionally Ollama.

## 1. Install Python 3.12
1. Download **Python 3.12 (64-bit)** from https://www.python.org/downloads/windows/.
2. In the installer, tick **"Add python.exe to PATH"**, then click **Install Now**.
3. Open **Command Prompt** and check: `py -3.12 --version`. It should print `Python 3.12.x`.

## 2. Install Tesseract OCR (reads text and numbers on screen)
1. Download the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki (`tesseract-ocr-w64-setup-*.exe`).
2. Install it to the default folder `C:\Program Files\Tesseract-OCR`.
3. Check: `"C:\Program Files\Tesseract-OCR\tesseract.exe" --version`.

## 3. Get the two free API keys
1. **Groq** (text answers): sign in at https://console.groq.com/keys, choose **Create API Key**, and copy the key (`gsk_...`).
2. **OpenRouter** (screen and chart images): sign in at https://openrouter.ai/settings/keys, choose **Create Key**, and copy it (`sk-or-...`). Free `:free` models need no credit. If you add credit later, also set a spending limit on the key.

## 4. Unzip and set up the project
```bat
cd %USERPROFILE%\Desktop
:: unzip PowerAgent_Live_Ultimate.zip here, then:
cd PowerAgent_Live_Ultimate
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```
In Notepad, paste your keys after `GROQ_API_KEY=` and `OPENROUTER_API_KEY=`. If Tesseract is installed elsewhere, fix `TESSERACT_CMD`. Then save the file.

## 5. Check the setup (free)
```bat
python -m scripts.check_setup --no-calls
python -m scripts.check_setup
```
- The first command checks the keys, every model ID, Tesseract and your monitors.
- The second command makes one tiny call per role: master, fast, verifier and vision. With the free profile it costs $0.
- Lines marked ✓ are fine. Any ✗ line tells you what to fix.

## 6. Run the app
```bat
streamlit run app.py
```
The browser opens at http://127.0.0.1:8501. The app listens on this computer only. Next time, double-click **run_windows.bat**, which creates the virtual environment the first time and then starts the app.

## 7. Use it
1. **Live Monitor**: either click **🖥️ Share a tab** in the sidebar and pick the browser tab with your dashboard (this works locally and online), or, on a local run, choose a monitor, region or window under *Capture this computer* and press **Start monitoring**. A capture is taken every 2 s by default. OCR and the chart reader turn each frame into timestamped estimated values.
2. **Data**: upload CSV or XLSX files. These are verified values, and the app prefers them over screen estimates.
3. **Chat**: ask questions, for example:
   - "What is on my screen?"
   - "Which metric is changing fastest?"
   - "Compare the last 10 minutes with the previous 10"
   - "Why did current_a rise?"
   - "What should I do?"
   - "Did your suggestion work?"

   Each answer carries:
   - a verdict pill: Verified, Partly verified, or Check the numbers;
   - `[REF-n]` citation chips;
   - the provider and model, tokens and time;
   - an **Evidence and agent activity** panel.
4. **Analytics**, **Alerts**, **History**, **Settings** and **Diagnostics** work as in the previous version. Settings now has the model profile and the image-answer switch.

## 7b. Vision-first: how image inferences work
- **Background reading:** every changed frame (tab share, local capture or an uploaded image) is read by the vision model, at most once per `VISION_NOTES_INTERVAL` (60 s). The model is free Gemma 4 first, then free Qwen 2.5 VL, then any other free Gemma or Qwen vision model OpenRouter lists. Each reading records:
  - every chart's title, axes, series, trend, peaks, dips, threshold crossings and anomalies;
  - KPI values and alarm indicators;
  - findings marked info, warning or critical, plus an overall assessment.
- **Where to see it:** **Live Monitor → 🧠 AI reading of the screen** shows the latest reading. **🧠 Read the screen with AI now** forces a new one.
- **Readings build a history.** Values read by the vision model become the `vision_live` data source, so trends, anomalies and alerts work on them over time. A value that also appears in the OCR text is marked as confirmed by OCR.
- **Chat:** questions about the screen send the whole screen plus the largest chart crops, together with the latest reading and the previous readings. The answer has four parts: *What the screen shows*, *Inferences*, *Abnormal / attention* and *What to check next*.
- **Uploading an image:** use the 📎 in the chat box to attach a screenshot, a photo of a panel or a chart image. It is analysed the same way.
- **Free-model limits:**
  - OpenRouter limits free models per minute and per day. The daily limit is much higher once your account has bought credits; see openrouter.ai/docs for the current numbers.
  - After a rate-limit error the app skips that model for 60 s, tries the next one, and slows background reading (1 → 15 min).
  - For heavy use, raise `VISION_NOTES_INTERVAL` or ask questions only on demand.

## 8. Which model answers what (same as RAG∞ Ultimate)
| Role | Free profile chain | Used for |
|---|---|---|
| master | Groq `gpt-oss-120b` → `llama-3.3-70b-versatile` → OpenRouter `gemma-4-31b-it:free` | Cited answers. The model may call analysis tools for why, compare and advice questions. |
| fast | Groq `gpt-oss-20b` → `llama-3.1-8b-instant` | Small jobs |
| verifier | Groq `llama-3.3-70b` → `gpt-oss-20b` | Optional LLM fact-check (`VERIFY_WITH_LLM`) |
| vision | OpenRouter `gemma-4-31b-it:free` → `qwen2.5-vl-72b-instruct:free` → `gemma-4-26b-a4b-it:free` → `qwen2.5-vl-32b-instruct:free` → other free Gemma/Qwen vision models found automatically | Reading every changed frame and answering image questions |

- With `MODEL_PROFILE=quality`, Qwen 3.8 Flash reads images first. It costs a few cents.
- When a model is rate-limited, down or out of credit, the router moves to the next one in the chain.

## 9. Privacy: what leaves your PC
- **Text evidence goes to Groq.** This is the computed numbers plus OCR text, never the raw screenshot.
- **Images go to OpenRouter only for a visual question.** They are chart crops, sent only when the question is about what the screen shows and `IMAGE_ANSWERS=true`.
- **Background vision notes are off by default.** Turn them on with `VISION_NOTES=true`.
- **To keep everything on your PC**, install Ollama (step 10) and set `MODEL_PROFILE=local` and `LOCAL_ONLY=true`.
- **Screen and file text is treated as data.** Instruction-like lines are neutralised by the prompt-injection guard. The model has no shell, code or file tools, only the analysis functions.

## 10. Optional: fully offline with Ollama
1. Install Ollama from https://ollama.com/download.
2. Pull the models:
   ```bat
   ollama pull qwen2.5:7b
   ollama pull llama3.2:3b
   ollama pull qwen2.5vl:7b
   ```
3. In `.env`, set `MODEL_PROFILE=local` and `LOCAL_ONLY=true`. Alternatively, keep the cloud profile and set `OLLAMA_ENABLED=true` to use Ollama as the last fallback.
4. Run `python -m scripts.check_setup` again.

## 11. Optional: chart model and ChartNet benchmark
- See `chart_model_server/README.md`. It runs Granite Vision 4.1 (Apache-2.0, trained with ChartNet) as a local server at http://127.0.0.1:8600.
- Set `CHART_MODEL_ENABLED=true` to use it.
- The ChartNet benchmark runs from **Diagnostics**, or with:
  ```bat
  python -m metrics.chartnet_eval --samples 200
  ```

## 12. Put it online like RAG∞ (Streamlit Community Cloud, free)
1. Create a **private** GitHub repo and upload the project folder. `.env` and `data/` are ignored by `.gitignore`, so your keys stay off GitHub.
2. Go to https://share.streamlit.io, choose **Create app**, pick the repo, set **Main file** to `app.py`, and open **Advanced settings**.
3. Choose **Python 3.12**, and paste your secrets:
   ```toml
   GROQ_API_KEY = "gsk_..."
   OPENROUTER_API_KEY = "sk-or-..."
   APP_PASSWORD = "choose-a-password"
   MODEL_PROFILE = "free"
   ```
4. Click **Deploy**.
   - `packages.txt` installs Tesseract on the server automatically.
   - `requirements.txt` installs the Python packages.
5. Open the app URL in Chrome or Edge on a computer.
   - Click **🖥️ Share a tab** in the sidebar and pick the tab or window with your live dashboard.
   - A frame goes to the server every 2 s. It is read with OCR and the chart reader, and sharing keeps running while you chat.
   - Click **■ Stop sharing**, or close the browser's sharing bar, to stop.

Things to know about the hosted app:
- **The server cannot see your screen by itself.** Only the browser share works there; local capture (mss) is for local runs. A local run shows both options.
- **Everyone who opens the URL shares one data space and one chat history.** Set `APP_PASSWORD`, and don't share the link publicly.
- **The server's disk is temporary.** History, uploads and alerts are lost when Streamlit restarts the app. For permanent history, run it locally or with Docker and a volume: `docker build -t poweragent . && docker run -p 8501:8501 -v poweragent:/data --env-file .env poweragent`.
- **Frames are processed only in memory on the server,** unless you turn on `STORE_SCREENSHOTS`. The same privacy rules as step 9 apply.

## 13. Tests (optional)
```bat
pip install -r requirements-dev.txt
python -m pytest -q
```
The tests run offline with a scripted model router; no keys are needed.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "No AI model is configured" | Put `GROQ_API_KEY` in `.env` and restart the app. |
| Answers say "Computed results (no AI)" | A key is wrong, or the Groq limit was reached. Run `python -m scripts.check_setup`. The computed numbers are still correct. |
| "Tesseract is not installed" | Do step 2, or fix `TESSERACT_CMD` in `.env`. |
| Screen questions don't use images | Set `OPENROUTER_API_KEY` and `IMAGE_ANSWERS=true`, and capture the screen first. |
| `'qwen2.5:7b' is not installed in Ollama` | Run `ollama pull qwen2.5:7b`. |
| Black capture of a window | Some apps block capture. Capture the monitor or a region instead. |
