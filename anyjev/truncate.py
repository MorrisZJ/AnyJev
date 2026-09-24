"""Write a model with its last blocks removed, so every runtime can serve a truncated forward.

A decision does not need the whole model. On typed-decisions a closed-form head is flat from
about two thirds of the depth, and the blocks above that are converting the answer into token
space rather than deciding anything. Skipping them is the one efficiency lever that is linear
in cost and costs almost no accuracy -- but only the local `transformers` path can stop a
forward early, and an inference server cannot be asked to run "most of" a model.

So do it on disk instead. `truncate(model, blocks, out)` writes a normal checkpoint that simply
has fewer layers: `config.json` with `num_hidden_layers = blocks`, the tensors of those blocks,
the embeddings, the final norm and the head. Nothing downstream has to know. vLLM serves it,
`transformers` loads it, a quantiser quantises it, llama.cpp converts it -- because it is not a
special object, it is a smaller model.

    python -m anyjev.truncate Qwen/Qwen2.5-7B-Instruct 18 /models/qwen2.5-7b-b18

**The head must be fit on the model it is served by.** A truncated model applies its final norm
after the last block it kept, so its output vector is `norm(h_b)`, while `HFBackend` captures
`h_b` itself at an intermediate block. Those are different spaces, and a head moved between
them silently reads the wrong one. Fit against whichever artifact you will deploy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Optional

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# small files a served model needs next to its weights
_SIDECAR = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "tokenizer.model", "generation_config.json",
            "chat_template.jinja", "preprocessor_config.json")


def layer_of(name: str) -> Optional[int]:
    """The block index a tensor belongs to, or None for embeddings, the final norm and the head."""
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def truncate(model: str, blocks: int, out: str, *, dtype: Optional[str] = None,
             overwrite: bool = False) -> str:
    """Write `model`'s first `blocks` layers to `out` as a standalone checkpoint. Returns `out`."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    src = model if os.path.isdir(model) else snapshot_download(model)
    cfg = AutoConfig.from_pretrained(src)
    total = int(getattr(cfg, "num_hidden_layers", 0))
    if not 1 <= blocks <= total:
        raise ValueError(f"blocks must be in 1..{total} for {model}, got {blocks}")

    torch_dtype = getattr(torch, dtype) if dtype else "auto"
    m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch_dtype, device_map="cpu")
    state = m.state_dict()
    keep: Dict[str, "torch.Tensor"] = {}
    dropped: List[str] = []
    for name, tensor in state.items():
        li = layer_of(name)
        if li is not None and li >= blocks:
            dropped.append(name)
            continue
        keep[name] = tensor.contiguous()

    os.makedirs(out, exist_ok=True)
    save_file(keep, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    cfg.num_hidden_layers = blocks
    cfg.save_pretrained(out)
    # record where this came from: a truncated model is not the model it is named after
    meta = {"anyjev_truncated_from": model, "kept_blocks": blocks, "original_blocks": total,
            "dropped_tensors": len(dropped)}
    with open(os.path.join(out, "anyjev_truncation.json"), "w") as f:
        json.dump(meta, f, indent=2)
    try:
        AutoTokenizer.from_pretrained(src).save_pretrained(out)
    except Exception:  # noqa: BLE001 - fall back to copying whatever sidecar files exist
        for fn in _SIDECAR:
            p = os.path.join(src, fn)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(out, fn))
    print(f"{model}: kept {blocks}/{total} blocks, dropped {len(dropped)} tensors -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("blocks", type=int)
    ap.add_argument("out")
    ap.add_argument("--dtype", default=None, help="bfloat16 / float16; default: the checkpoint's")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    truncate(args.model, args.blocks, args.out, dtype=args.dtype, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
