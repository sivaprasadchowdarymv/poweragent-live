# Chart model server — Granite Vision on Windows (free)

The Live Dashboard Agent can use a chart-specialised AI model as a **second chart reader** alongside its built-in pixel reader. Everything here is free and runs on your own PC:

| Part | What | Licence |
|---|---|---|
| **Granite Vision 4.1 (4B)** — recommended | IBM's chart/table/document vision model, trained with the ChartNet dataset; has a built-in `<chart2csv>` task | Apache-2.0 |
| ChartAst-S (13B) — alternative | OpenGVLab chart model; needs a large GPU and WSL2; see [CHARTAST.md](CHARTAST.md) | Apache-2.0 (HF page) + LLaMA 2 licence |
| **ChartNet** dataset | Used by the app's benchmark to *measure* each reader's accuracy | `core_permissive`: CDLA-Permissive-2.0 |

How it is used in the app:

1. Each chart the pixel reader finds is cropped and sent to the server (`<chart2csv>` → CSV table).
2. The table is parsed, then **cross-checked** point by point against the pixel reader (✅ agree / 🟡 partial / ⚠️ disagree).
3. Values are stored as a separate *estimated* source, `screen_chart_model`, and are never merged with other data.
4. In the chat, *"according to the chart …"* questions are answered by the model and labelled as estimates.
5. If you have run the **ChartNet benchmark**, each reader's live confidence is scaled by its measured accuracy for that chart type.

## Hardware guide (estimates — not measured here)

| Your PC | How to run | Expect |
|---|---|---|
| NVIDIA GPU with ≥ 12 GB | `python granite_server.py` (bf16, about 9 GB of VRAM) | a few seconds per chart |
| NVIDIA GPU with 6–8 GB | `python granite_server.py --quant 4bit` (needs bitsandbytes) | a few seconds per chart, slightly lower accuracy |
| No NVIDIA GPU, ≥ 16 GB RAM | `python granite_server.py --device cpu` (about 9–10 GB RAM) | slow: often a minute or more per chart; set the app's chart-model interval to 120 s or more |

The weights (about 8 GB) download once from Hugging Face. No account or payment is needed, and after the first download you can run with `--offline`.

## Install on Windows (PowerShell)

Use a **separate** virtual environment so the heavy AI packages don't affect the app.

**1. Create the environment** (from the project folder, e.g. `D:\Projects\PowerAgent_Live_Dashboard_Agent`):
```powershell
cd D:\Projects\PowerAgent_Live_Dashboard_Agent\chart_model_server
py -3.12 -m venv .venv-granite
.\.venv-granite\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**2. Install PyTorch.** Choose one:
```powershell
# NVIDIA GPU (check first that `nvidia-smi` works):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# no NVIDIA GPU:
pip install torch
```
Check it:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**3. Install the model libraries:**
```powershell
pip install -r requirements-granite.txt
pip install bitsandbytes          # only if you will use --quant 4bit
```

**4. Start the server.** The first run downloads about 8 GB:
```powershell
python granite_server.py                  # GPU with >= 12 GB
python granite_server.py --quant 4bit     # GPU with 6-8 GB
python granite_server.py --device cpu     # no GPU
```
Wait for `Serving granite-vision-4.1-4b on http://127.0.0.1:8600`, and keep this window open.

**5. Check it.** Open <http://127.0.0.1:8600/health> in your browser. It should show `"ok": true`, the device, and `"output_format": "csv"`.

**6. Turn it on in the app.** Go to **Settings** → *Chart model* → **Use chart model** on, URL `http://127.0.0.1:8600`, interval `30` (use `120` on CPU) → **Apply settings** → **Test chart model server**.

**7. Use it.**
* **Live Monitor** → *Chart-model readings*: the table read for each chart, the model name, and the cross-check result. **Read charts with the chart model now** reads immediately.
* Chat: *"What is happening on my screen?"* and *"According to the chart, which month had the highest load?"*

## Measure accuracy with ChartNet (free)

ChartNet pairs every chart with its true data table, so the app can score each reader on real chart variety. Only the sampled rows are downloaded.

```powershell
cd D:\Projects\PowerAgent_Live_Dashboard_Agent
.\.venv\Scripts\Activate.ps1                      # the app's environment
pip install -r live_dashboard_agent\requirements-benchmark.txt
python -m live_dashboard_agent.benchmarks.chartnet_eval --samples 200                         # built-in pixel reader only
python -m live_dashboard_agent.benchmarks.chartnet_eval --samples 200 --readers pixel,model   # with the chart model server running
```

You can also run it from the app: **Diagnostics** → *ChartNet accuracy benchmark* → **Run ChartNet benchmark**, then **Use these results for confidence**.

The report (`report.md`, `results.csv`, `summary.json`, `priors.json`) is saved under the app's data folder in `benchmarks\<timestamp>\`. It shows extraction rate, recall, precision and F1 within 5 % and 10 % of the true values, overall and per chart type.

Notes:
* The default subset `core_permissive` is CDLA-Permissive-2.0, which is free, including commercial use. `--subset human_verified` (2,000 human-checked test charts) is free to download but carries IBM's notice against commercial exploitation or deployment, so use it for private evaluation only. The report flags this.
* The pixel reader is designed for coloured line, area and bar charts with numeric axes. Pie, heatmap, box, violin and scatter-type charts score low by design; the benchmark shows exactly where it is reliable.
* If Hugging Face asks you to log in or accept terms for the dataset, create a free account and run `huggingface-cli login`.

## API

`GET /health` · `POST /v1/chart_to_table {"image": "<base64>"}` · `POST /v1/chart_qa {"image", "question"}` · `POST /v1/summarize {"image"}`. The server is the same for Granite and ChartAst (`common.py`). It runs one request at a time and listens on `127.0.0.1` only by default.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA: False` with an NVIDIA GPU | Update the NVIDIA driver, then reinstall torch with the `cu128` command |
| `CUDA out of memory` | Use `--quant 4bit`, or close other GPU programs |
| Error mentioning `trust_remote_code` or an unknown architecture | Upgrade: `pip install -U transformers`; with older versions add `--trust-remote-code` |
| Very slow on CPU | Expected. Raise the app's chart-model interval, or use a GPU |
| App says the server can't be reached | Keep the server window open; check http://127.0.0.1:8600/health |
