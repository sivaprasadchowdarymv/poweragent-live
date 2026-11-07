"""ChartAst (ChartAssistant, OpenGVLab) inference server.

Runs on a machine with an NVIDIA GPU, inside a clone of
https://github.com/OpenGVLab/ChartAst, and exposes a small local HTTP API that
the Live Dashboard Agent calls:

    GET  /health
    POST /v1/chart_to_table   {"image": "<base64 PNG/JPEG>"}
    POST /v1/chart_qa         {"image": "...", "question": "..."}
    POST /v1/summarize        {"image": "..."}

Model loading, image preprocessing and prompts follow the repo's own
evaluation script (accessory/single_turn_eval_multitask.py):
  * MetaModel("llama_ens5", params.json, tokenizer.model, with_visual=True)
  * load_tensor_parallel_model_list(model, <dir with consolidated.*.model.pth>)
  * PadToSquare(CLIP mean) -> Resize(448, bicubic) -> ToTensor -> CLIP Normalize
  * Alpaca prompt "### Instruction:\\n<instruction>\\n\\n### Response:",
    answers cut at the first "###"
  * optional 4-bit NF4 quantization via accessory.util.quant.quantize (as in the repo's demos)

Usage (see chart_model_server/CHARTAST.md):
    python chartast_server.py --repo /path/to/ChartAst --weights /path/to/ChartAssistant \
                     --host 127.0.0.1 --port 8600 [--quant 4bit]

The HTTP layer is shared with the Granite Vision server (common.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import serve  # noqa: E402

log = logging.getLogger("chart_model_server")

ALPACA = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
INSTRUCTIONS = {
    "table": "convert this chart to a table.",
    "qa": "Please answer my question based on the chart: {question}",
    "summary": "Please summarize the chart.",
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_prompt(task: str, question: str = "") -> str:
    instruction = INSTRUCTIONS[task].format(question=question.strip())
    return ALPACA.format(instruction=instruction)


def clean_answer(text: str) -> str:
    """Same post-processing as the repo: keep text before the first '###'."""
    return (text or "").split("###")[0].strip()


class ChartAstEngine:
    """Loads ChartAst-S exactly like the repository's evaluation script."""

    name = "ChartAst-S"
    output_format = "markdown"

    def __init__(self, repo: str, weights: str, quant: str = "none", input_size: int = 448, max_seq_len: int = 4096, top_p: float = 0.5) -> None:
        self.top_p = top_p
        import torch
        import torchvision.transforms as T
        from PIL import Image

        repo = os.path.abspath(repo)
        for p in (repo, os.path.join(repo, "accessory")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from accessory.model.meta import MetaModel  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            raise RuntimeError("ChartAst needs an NVIDIA GPU with CUDA; torch.cuda.is_available() is False.")
        shards = sorted(f for f in os.listdir(weights) if f.startswith("consolidated.") and f.endswith(".model.pth"))
        if not shards:
            raise FileNotFoundError(f"No consolidated.*.model.pth files in {weights}. Download FanqingM/ChartAssistant first.")
        cfg = os.path.join(weights, "params.json") if os.path.exists(os.path.join(weights, "params.json")) else os.path.join(repo, "params.json")
        tok = os.path.join(weights, "tokenizer.model") if os.path.exists(os.path.join(weights, "tokenizer.model")) else os.path.join(repo, "tokenizer.model")
        t0 = time.time()
        # The repo's own loader: builds the model directly in bf16 on the GPU (or on CPU then
        # quantizes to 4-bit NF4), initializes single-process distributed mode, loads both shards.
        model = MetaModel.from_pretrained(
            weights,
            llama_type="llama_ens5",
            llama_config=[cfg],
            tokenizer_path=tok,
            with_visual=True,
            max_seq_len=max_seq_len,
            dtype=torch.bfloat16,
            device="cuda",
            quant=(quant == "4bit"),
        )
        model.eval()
        self.model = model
        self.torch = torch

        class PadToSquare:
            def __call__(self, img: Image.Image) -> Image.Image:
                w, h = img.size
                if w == h:
                    return img
                side = max(w, h)
                bg = tuple(int(c * 255) for c in CLIP_MEAN)
                out = Image.new(img.mode, (side, side), bg)
                out.paste(img, ((side - w) // 2, (side - h) // 2))
                return out

        self.transform = T.Compose([PadToSquare(), T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
        self.info = {
            "model": self.name,
            "quant": quant,
            "input_size": input_size,
            "device": torch.cuda.get_device_name(0),
            "load_seconds": round(time.time() - t0, 1),
            "gpu_memory_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }

    def run(self, task: str, image: Any, question: str, max_new_tokens: int, temperature: float) -> str:
        return clean_answer(self.generate(build_prompt(task, question), image, max_new_tokens, temperature, self.top_p))

    def generate(self, prompt: str, image: Any, max_new_tokens: int, temperature: float, top_p: float) -> str:
        torch = self.torch
        x = self.transform(image.convert("RGB")).unsqueeze(0).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.model.generate([prompt], x, max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        return out[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="ChartAst inference server for the Live Dashboard Agent")
    ap.add_argument("--repo", required=True, help="Path to the cloned OpenGVLab/ChartAst repository")
    ap.add_argument("--weights", required=True, help="Directory with consolidated.00-of-02.model.pth and consolidated.01-of-02.model.pth")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default localhost only)")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="4bit = NF4 via bitsandbytes (less GPU memory, slightly lower accuracy)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic (recommended for reading values)")
    ap.add_argument("--top-p", type=float, default=0.5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Loading ChartAst-S (this takes a few minutes the first time)…")
    engine = ChartAstEngine(args.repo, args.weights, args.quant, top_p=args.top_p)
    log.info("Loaded: %s", engine.info)
    serve(engine, args.host, args.port, args.temperature)


if __name__ == "__main__":
    main()
