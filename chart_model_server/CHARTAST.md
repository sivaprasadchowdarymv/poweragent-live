# ChartAst server — setup on a GPU machine (alternative to Granite Vision)

> For most Windows PCs, **Granite Vision 4.1** (README.md in this folder) is the easier and lighter choice. ChartAst needs a much larger GPU.

[ChartAst / ChartAssistant](https://github.com/OpenGVLab/ChartAst) (OpenGVLab, ACL 2024) is a chart-specialised vision-language model trained on chart-to-table translation and chart question answering. The Live Dashboard Agent uses it as an **optional second chart reader**:

* every detected chart is cropped and sent to ChartAst with the instruction `convert this chart to a table.`;
* the returned Markdown table is parsed into numeric series;
* those numbers are **cross-checked** against the app's pixel-calibration reading of the same chart. Agreement raises confidence to 0.9; disagreement lowers it to 0.35 and is shown with a ⚠️;
* the values are stored as a separate *estimated* source, `screen_chart_model`, and are never merged with the pixel reader or with file data;
* in the chat, *"according to the chart …"* questions are also sent to ChartAst (`Please answer my question based on the chart: …`).

`chartast_server.py` loads the model with the repository's own code (`MetaModel.from_pretrained`, `llama_type="llama_ens5"`). It uses the same preprocessing as the repo's evaluation script (pad to square → 448 px bicubic → CLIP normalisation) and the same Alpaca prompts, and cuts the answer at the first `###`.

## What you need

| | Requirement |
|---|---|
| Model | Only **ChartAst-S** (13B, LLaMA-2 + SPHINX visual encoders) is published: `FanqingM/ChartAssistant`, 2 files × 19.9 GB = **39.8 GB**. The small ChartAst-D (260M) was never released. |
| OS | **Linux, or Windows through WSL2** (Ubuntu). The repo initialises PyTorch distributed with the NCCL backend, which native Windows does not support. |
| GPU | NVIDIA with CUDA. **Estimates, not measured here:** about 35–40 GB VRAM in bf16 (A100 40 GB, A6000/L40S 48 GB); with `--quant 4bit`, roughly 14–20 GB (an RTX 3090/4090 24 GB is the realistic minimum). |
| System RAM | 64 GB recommended. In 4-bit mode the model is built in bf16 on the CPU before quantisation. |
| Disk | About 60 GB free (weights plus Python environment). |

> I could not run the real model while building this integration (no GPU was available). The server's HTTP layer, the app adapter, the table parser and the cross-check are covered by automated tests using a stub engine. Treat the memory figures above as estimates and check `/health` (it reports GPU memory used) on your machine.

## Step by step (Windows + WSL2)

**1. WSL2 with GPU support.** Install the current NVIDIA Windows driver. Then, in PowerShell as administrator:
```powershell
wsl --install -d Ubuntu-22.04
```
Open *Ubuntu* and check that the GPU is visible:
```bash
nvidia-smi
```

**2. Python environment (inside Ubuntu):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash && exec bash
conda create -n chartast python=3.10 -y
conda activate chartast
```

**3. Get the ChartAst code and its dependencies:**
```bash
git clone https://github.com/OpenGVLab/ChartAst.git ~/ChartAst
cd ~/ChartAst
pip install -r requirements.txt            # torch 2.0.1 + CUDA 11.7 wheels, fairscale, open_clip, bitsandbytes, ...
cp Readme.md README.md                     # setup.py expects README.md (case-sensitive on Linux)
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If `torch.cuda.is_available()` is `False`, install a torch build that matches your driver (see pytorch.org), for example `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`.

**4. Download the weights (39.8 GB):**
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download FanqingM/ChartAssistant --local-dir ~/models/ChartAssistant
ls -lh ~/models/ChartAssistant             # consolidated.00-of-02.model.pth, consolidated.01-of-02.model.pth, params.json, tokenizer.model
```

**5. Start the server.** Run it straight from the project folder (it needs `common.py` next to it); from WSL your Windows drives are under `/mnt/c/...` or `/mnt/d/...`:
```bash
cd ~/ChartAst
python /mnt/d/Projects/PowerAgent_Live_Dashboard_Agent/chart_model_server/chartast_server.py \
       --repo ~/ChartAst --weights ~/models/ChartAssistant --port 8600            # add --quant 4bit for 24 GB GPUs
```
Loading takes a few minutes. The first start needs internet once, because `torch.hub` fetches the DINOv2 model *code* from GitHub (no extra weights are downloaded). Wait for `Serving on http://127.0.0.1:8600`.

**6. Check it from Windows.** Open <http://127.0.0.1:8600/health> in a browser; it should show `"ok": true`, the GPU name and the memory used. WSL2 forwards `localhost` to Windows by default. If it doesn't, either enable mirrored networking (`networkingMode=mirrored` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`), or start the server with `--host 0.0.0.0` and use the WSL IP address (that also requires turning off local-only mode in the app).

**7. Turn it on in the app.** Go to **Settings** → *Chart model* → **Use chart model** on, URL `http://127.0.0.1:8600`, interval 30 s → **Apply settings** → **Test chart model server**.

You can also set it before launching:
```powershell
$env:LDA_CHART_MODEL_ENABLED = "true"
$env:LDA_CHART_MODEL_URL = "http://127.0.0.1:8600"
```

**8. Use it.**
* **Live Monitor** → *Chart-model readings*: the table read for each chart, with the cross-check result (✅ agree / 🟡 partial / ⚠️ disagree) and inference time. **Read charts with the chart model now** triggers a reading immediately.
* Chat: *"What is happening on my screen?"* now includes the ChartAst tables. *"According to the chart, which month was highest?"* is answered by ChartAst and labelled as a model estimate.
* Data Sources → *Metrics with more than one source*: compare `screen_live` (pixel reader) with `screen_chart_model` and choose which one answers questions.

## Using a GPU on another computer

Keep the server bound to `127.0.0.1` on the GPU machine and tunnel to it:
```powershell
ssh -L 8600:127.0.0.1:8600 user@gpu-box
```
The app then still talks to `http://127.0.0.1:8600` with local-only mode on, and screenshots never travel unencrypted. The server has **no authentication**. Do not bind it to `0.0.0.0` on an untrusted network.

## API (for reference)

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | — | `ok`, `busy`, model, quantisation, device, GPU memory, request counts |
| `POST /v1/chart_to_table` | `{"image": "<base64 PNG/JPEG>"}` | `{"text": "<markdown table>", "ms": …}` |
| `POST /v1/chart_qa` | `{"image": …, "question": "…"}` | `{"text": "<answer>", "ms": …}` |
| `POST /v1/summarize` | `{"image": …}` | `{"text": "<summary>", "ms": …}` |

The server runs one generation at a time (GPU lock) and returns HTTP 503 if busy for longer than 5 minutes. Generation is deterministic by default (`--temperature 0`), and images are limited to 12 MB.

## Licence

The Hugging Face model page lists Apache-2.0. ChartAst-S is built on LLaMA 2, and the repository ships `LICENSE_llama2`, so check both for your use case.
