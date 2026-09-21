"""transformers backend: single forward per prompt batch, logits at the last position."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class HFBackend:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 16, trust_remote_code: bool = False, revision: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code,
                                                       revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=trust_remote_code, revision=revision)
        self.model.eval()

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        import torch

        n = len(prompts)
        # sort by length to reduce padding, restore order at the end
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        order = sorted(range(n), key=lambda i: lengths[i])
        out: List[Optional[np.ndarray]] = [None] * n
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt",
                                 padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            # explicit position ids so left padding does not shift positions
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = self.model(**enc, position_ids=pos).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            for row, i in enumerate(idx):
                ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                out[i] = lp[row, ids].cpu().numpy().astype(np.float64)
        return out  # type: ignore[return-value]
