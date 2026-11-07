"""Granite Vision chart server (IBM, Apache-2.0) — runs natively on Windows.

IBM's Granite Vision 4.1 (4B) was trained with the ChartNet dataset and has
dedicated chart tasks. This server uses them exactly as the model card shows:
a user message with an image block and the task tag as text
(``<chart2csv>``, ``<chart2summary>``), formatted with the processor's chat
template. Chart questions are sent as plain text with the image.

    python granite_server.py                     # GPU if available, else CPU
    python granite_server.py --quant 4bit        # ~4-bit weights on an NVIDIA GPU (needs bitsandbytes)
    python granite_server.py --device cpu        # force CPU (slow, but works without a GPU)

The weights (~8 GB, BF16) are downloaded once from Hugging Face into the
normal cache; afterwards the server works offline (add --offline).
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

DEFAULT_MODEL = "ibm-granite/granite-vision-4.1-4b"
TASK_TAGS = {"table": "<chart2csv>", "summary": "<chart2summary>"}


def clean_output(text: str) -> str:
    """Strip a surrounding ```csv ... ``` fence and whitespace."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def build_messages(task: str, question: str) -> list[dict[str, Any]]:
    prompt = TASK_TAGS.get(task) or question.strip()
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


class GraniteVisionEngine:
    output_format = "csv"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "auto", quant: str = "none", max_side: int = 1536, offline: bool = False, trust_remote_code: bool = False) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoModel
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoModel  # type: ignore[no-redef]

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but PyTorch cannot see a CUDA GPU. Install the CUDA build of PyTorch or use --device cpu.")
        dt = {"auto": torch.bfloat16, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        kwargs: dict[str, Any] = {"device_map": dev}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        if quant == "4bit":
            if dev != "cuda":
                raise RuntimeError("--quant 4bit needs an NVIDIA GPU.")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        t0 = time.time()
        log.info("Loading %s on %s (%s)…", model_id, dev, "4-bit" if quant == "4bit" else str(dt).replace("torch.", ""))
        try:
            model = AutoModel.from_pretrained(model_id, dtype=dt, **kwargs)  # transformers >= 4.56
        except TypeError:
            model = AutoModel.from_pretrained(model_id, torch_dtype=dt, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, **({"trust_remote_code": True} if trust_remote_code else {}))
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model = model.eval()
        self.torch = torch
        self.max_side = max_side
        self.name = model_id.split("/")[-1]
        self.info = {
            "model": self.name,
            "model_id": model_id,
            "trained_with": "ChartNet (IBM) per the Granite Vision 4.x release",
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "CPU",
            "dtype": "4bit-nf4" if quant == "4bit" else str(dt).replace("torch.", ""),
            "quant": quant,
            "load_seconds": round(time.time() - t0, 1),
            "gpu_memory_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2) if dev == "cuda" else 0,
            "license": "Apache-2.0",
        }

    def run(self, task: str, image: Any, question: str, max_new_tokens: int, temperature: float) -> str:
        torch = self.torch
        img = image.convert("RGB")
        if max(img.size) > self.max_side:
            img.thumbnail((self.max_side, self.max_side))
        text = self.processor.apply_chat_template(build_messages(task, question), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True).to(self.model.device)
        if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        gen: dict[str, Any] = {"max_new_tokens": max_new_tokens, "use_cache": True}
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature)
        else:
            gen["do_sample"] = False
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        decoded = self.processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return clean_output(decoded)


def main() -> None:
    ap = argparse.ArgumentParser(description="Granite Vision chart server for the Live Dashboard Agent (free, Apache-2.0)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id or local folder (default {DEFAULT_MODEL})")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="auto = bfloat16 (also on CPU, to halve RAM)")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="4bit needs an NVIDIA GPU and bitsandbytes")
    ap.add_argument("--max-side", type=int, default=1536, help="Downscale images larger than this (pixels)")
    ap.add_argument("--offline", action="store_true", help="Use only the local Hugging Face cache (after the first download)")
    ap.add_argument("--trust-remote-code", action="store_true", help="Needed only with older transformers versions")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = GraniteVisionEngine(args.model, args.device, args.dtype, args.quant, args.max_side, args.offline, args.trust_remote_code)
    log.info("Loaded: %s", engine.info)
    serve(engine, args.host, args.port)


if __name__ == "__main__":
    main()
