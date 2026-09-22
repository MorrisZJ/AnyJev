"""transformers backend for vision-language models: one forward per prompt
batch, logits at the last position, images attached to the prompts that
reference them.

Same contract as `hf.HFBackend` -- one prefill, nothing generated -- with two
differences that matter. The chat template lives on the processor rather than
the tokenizer, because only the processor knows how many tokens a picture
expands to. And position ids are left to the model: vision models with
multimodal rope derive them from the image grid, so the explicit-position
trick the text backend uses to survive left padding would corrupt them.

    Decider(VLMBackend("Qwen/Qwen3-VL-2B-Instruct"))
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

# rough tokens-per-image, used only to bucket prompts of similar length
_IMAGE_TOKEN_GUESS = 256


class VLMBackend:
    accepts_images = True

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 8, max_pixels: Optional[int] = None,
                 min_pixels: Optional[int] = None, trust_remote_code: bool = False,
                 revision: Optional[str] = None, attn_implementation: Optional[str] = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self.revision = revision
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        proc_kwargs = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = min_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision=revision, **proc_kwargs)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_renderer = self.processor   # the template that knows about images

        model_kwargs = {}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        # `dtype`, not `torch_dtype`: the vision path needs transformers >= 4.57, which renamed it
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=getattr(torch, dtype) if dtype != "auto" else "auto",
            device_map=device, trust_remote_code=trust_remote_code, revision=revision,
            **model_kwargs)
        self.model.eval()

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]],
                            images: Optional[Sequence[Sequence[Any]]] = None) -> List[np.ndarray]:
        import torch

        n = len(prompts)
        per_prompt = [tuple(x) for x in (images if images is not None else [()] * n)]
        if len(per_prompt) != n:
            raise ValueError(f"got {n} prompts but {len(per_prompt)} image lists")

        # sort by rough length to reduce padding, restore order at the end
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False))
                   + _IMAGE_TOKEN_GUESS * len(per_prompt[i]) for i, p in enumerate(prompts)]
        order = sorted(range(n), key=lambda i: lengths[i])
        out: List[Optional[np.ndarray]] = [None] * n
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            # the processor consumes one flat image list in prompt order and
            # matches it against the placeholders it finds in the text
            batch_images = [im.load() for i in idx for im in per_prompt[i]]
            enc = self.processor(text=[prompts[i] for i in idx],
                                 images=batch_images or None,
                                 return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            for row, i in enumerate(idx):
                ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                out[i] = lp[row, ids].cpu().numpy().astype(np.float64)
        return out  # type: ignore[return-value]
