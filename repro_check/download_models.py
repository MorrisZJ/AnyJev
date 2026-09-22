"""Pre-fetch every model the repro runs use.

Sequential on purpose: the HF CDN saturates one stream already, and a model
that fails (gated, missing weights) should not stop the others.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
]

IGNORE = ["*.pth", "*.gguf", "original/*", "*.onnx", "*consolidated*", "*.msgpack", "*.h5"]

for m in MODELS:
    t0 = time.time()
    print(f"### START {m}", flush=True)
    try:
        p = snapshot_download(m, ignore_patterns=IGNORE, max_workers=16)
        print(f"### DONE  {m} in {time.time() - t0:.0f}s -> {p}", flush=True)
    except Exception as e:
        print(f"### FAIL  {m} after {time.time() - t0:.0f}s: {type(e).__name__}: {e}", flush=True)

print("### ALL DOWNLOADS FINISHED", flush=True)
sys.stdout.flush()
