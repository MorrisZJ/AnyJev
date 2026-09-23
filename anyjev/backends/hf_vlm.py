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
        self.dtype = dtype
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
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.n_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        self.hidden_size = int(getattr(text_cfg, "hidden_size", 0))

    def _encode(self, prompts: Sequence[str], idx: Sequence[int], per_prompt: Sequence[tuple]):
        # the processor consumes one flat image list in prompt order and
        # matches it against the placeholders it finds in the text
        batch_images = [im.load() for i in idx for im in per_prompt[i]]
        enc = self.processor(text=[prompts[i] for i in idx], images=batch_images or None,
                             return_tensors="pt", padding=True, add_special_tokens=False)
        return enc.to(self.model.device)

    def _order(self, prompts: Sequence[str], per_prompt: Sequence[tuple]) -> List[int]:
        # sort by rough length to reduce padding; callers restore the order
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False))
                   + _IMAGE_TOKEN_GUESS * len(per_prompt[i]) for i, p in enumerate(prompts)]
        return sorted(range(len(prompts)), key=lambda i: lengths[i])

    def hidden_states(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None,
                      token_ids: Optional[Sequence[Sequence[int]]] = None,
                      positions: Optional[Sequence[Sequence[int]]] = None,
                      images: Optional[Sequence[Sequence[Any]]] = None):
        """The language model's hidden states at the last position, as `hf.HFBackend.hidden_states`
        returns them: (feats [N, len(layers), H] float32, lps, None). `layers` index the language
        model's hidden-state tuple (0 the embedding output, i block i, num_hidden_layers after the
        final norm). One full forward per batch: the positions of a vision model come from the
        image grid, so there is no early stop here. `positions` is not supported."""
        import torch

        if positions is not None:
            raise NotImplementedError("per-token features are not implemented for the vision backend")
        n = len(prompts)
        per_prompt = [tuple(x) for x in (images if images is not None else [()] * n)]
        if len(per_prompt) != n:
            raise ValueError(f"got {n} prompts but {len(per_prompt)} image lists")
        n_blocks = self.n_layers
        layers = [n_blocks] if layers is None else [(n_blocks + 1 + i) if i < 0 else int(i) for i in layers]
        feats = np.zeros((n, len(layers), self.hidden_size), dtype=np.float32)
        lps: List[Optional[np.ndarray]] = [None] * n
        order = self._order(prompts, per_prompt)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self._encode(prompts, idx, per_prompt)
            with torch.no_grad():
                try:
                    out = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
                except TypeError:
                    out = self.model(**enc, output_hidden_states=True)
            for li, layer in enumerate(layers):               # left padding: the last column is the last token
                feats[idx, li] = out.hidden_states[layer][:, -1, :].float().cpu().numpy()
            if token_ids is not None:
                lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                for row, i in enumerate(idx):
                    ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                    lps[i] = lp[row, ids].cpu().numpy().astype(np.float64)
            del out
        return feats, lps, None

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]],
                            images: Optional[Sequence[Sequence[Any]]] = None) -> List[np.ndarray]:
        import torch

        n = len(prompts)
        per_prompt = [tuple(x) for x in (images if images is not None else [()] * n)]
        if len(per_prompt) != n:
            raise ValueError(f"got {n} prompts but {len(per_prompt)} image lists")

        order = self._order(prompts, per_prompt)
        out: List[Optional[np.ndarray]] = [None] * n
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self._encode(prompts, idx, per_prompt)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            for row, i in enumerate(idx):
                ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                out[i] = lp[row, ids].cpu().numpy().astype(np.float64)
        return out  # type: ignore[return-value]
