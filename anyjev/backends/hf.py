"""transformers backend: single forward per prompt batch, logits at the last position."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class HFBackend:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 16, trust_remote_code: bool = False, revision: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.shared_fallbacks = 0   # groups whose prefix/suffix split was not token-exact
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

    def _last_logits(self, enc, pos):
        """Logits at the last position only. `logits_to_keep=1` skips the full-vocabulary
        projection for every earlier position (a 600-token prompt at batch 32 otherwise
        materializes ~10 GB of logits); older transformers without the kwarg fall back."""
        try:
            out = self.model(**enc, position_ids=pos, logits_to_keep=1)
        except TypeError:
            out = self.model(**enc, position_ids=pos)
        return out.logits[:, -1, :].float()

    # ---- optional fast path: one prefix forward per state, K short suffixes ----
    def _project(self, hidden):
        """Final logits for a [N, H] batch of last-position hidden states, matching
        what the CausalLM forward would return (including Gemma-style softcapping)."""
        import torch

        logits = self.model.lm_head(hidden).float()
        cap = getattr(self.model.config, "final_logit_softcapping", None)
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits

    def _repeat_cache(self, cache, repeats: int):
        """Expand a prefix KV cache from B rows to B*repeats rows (row b -> rows b*K..b*K+K-1)."""
        if hasattr(cache, "batch_repeat_interleave"):
            cache.batch_repeat_interleave(repeats)
            return cache
        from transformers import DynamicCache  # legacy tuple route for older versions

        legacy = tuple((k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0))
                       for k, v in cache.to_legacy_cache())
        return DynamicCache.from_legacy_cache(legacy)

    def score_shared(self, groups: Sequence[Tuple[str, Sequence[str]]],
                     token_ids: Sequence[Sequence[int]]) -> List[List[np.ndarray]]:
        """For each (prefix, suffixes) group, log p(token | prefix + suffix_j) for the
        group's token ids, computing the prefix once. Exact when the tokenizer splits
        prefix + suffix at the same boundary as the concatenation; groups where it does
        not are scored through the plain path (counted in `shared_fallbacks`)."""
        import torch

        tok = self.tokenizer
        results: List[List[Optional[np.ndarray]]] = [[None] * len(sfx) for _, sfx in groups]
        plan = []       # (group index, prefix ids, [suffix ids])
        fallback = []   # (group index, suffix index, full text)
        for gi, (prefix, suffixes) in enumerate(groups):
            p_ids = tok.encode(prefix, add_special_tokens=False)
            s_ids_list = [tok.encode(sfx, add_special_tokens=False) for sfx in suffixes]
            exact = all(tok.encode(prefix + sfx, add_special_tokens=False) == p_ids + s_ids
                        for sfx, s_ids in zip(suffixes, s_ids_list))
            if exact and len(suffixes) > 0:
                plan.append((gi, p_ids, s_ids_list))
            else:
                self.shared_fallbacks += 1
                fallback.extend((gi, si, prefix + sfx) for si, sfx in enumerate(suffixes))

        if fallback:
            lps = self.next_token_logprobs([t for _, _, t in fallback], [token_ids[gi] for gi, _, _ in fallback])
            for (gi, si, _), lp in zip(fallback, lps):
                results[gi][si] = lp

        pad = tok.pad_token_id
        device = self.model.device
        # batch groups of equal K together; B groups per forward so that B*K <= batch_size
        by_k: Dict[int, List[tuple]] = {}
        for entry in plan:
            by_k.setdefault(len(entry[2]), []).append(entry)
        for K, entries in by_k.items():
            B = max(1, self.batch_size // K)
            entries.sort(key=lambda e: len(e[1]))
            for start in range(0, len(entries), B):
                chunk = entries[start:start + B]
                b = len(chunk)
                Lp = max(len(e[1]) for e in chunk)
                p_input = torch.full((b, Lp), pad, dtype=torch.long)
                p_mask = torch.zeros((b, Lp), dtype=torch.long)
                for r, (_, p_ids, _) in enumerate(chunk):          # left-pad the prefixes
                    p_input[r, Lp - len(p_ids):] = torch.as_tensor(p_ids)
                    p_mask[r, Lp - len(p_ids):] = 1
                p_input, p_mask = p_input.to(device), p_mask.to(device)
                p_pos = (p_mask.cumsum(-1) - 1).clamp(min=0)
                with torch.no_grad():
                    p_out = self.model.model(input_ids=p_input, attention_mask=p_mask,
                                             position_ids=p_pos, use_cache=True)
                cache = self._repeat_cache(p_out.past_key_values, K)
                Ls = max(len(s_ids) for _, _, s_list in chunk for s_ids in s_list)
                s_input = torch.full((b * K, Ls), pad, dtype=torch.long)
                s_mask = torch.zeros((b * K, Ls), dtype=torch.long)
                last = torch.zeros(b * K, dtype=torch.long)
                for r, (_, _, s_list) in enumerate(chunk):        # right-pad the suffixes
                    for k, s_ids in enumerate(s_list):
                        row = r * K + k
                        s_input[row, :len(s_ids)] = torch.as_tensor(s_ids)
                        s_mask[row, :len(s_ids)] = 1
                        last[row] = len(s_ids) - 1
                s_input, s_mask, last = s_input.to(device), s_mask.to(device), last.to(device)
                full_mask = torch.cat([p_mask.repeat_interleave(K, 0), s_mask], dim=1)
                s_pos = p_pos[:, -1].repeat_interleave(K)[:, None] + 1 + torch.arange(Ls, device=device)[None, :]
                with torch.no_grad():
                    s_out = self.model.model(input_ids=s_input, attention_mask=full_mask, position_ids=s_pos,
                                             past_key_values=cache, use_cache=True)
                    hidden = s_out.last_hidden_state[torch.arange(b * K, device=device), last]
                    lp = torch.log_softmax(self._project(hidden), dim=-1)
                for r, (gi, _, s_list) in enumerate(chunk):
                    ids = torch.as_tensor(list(token_ids[gi]), device=device)
                    for k in range(len(s_list)):
                        results[gi][k] = lp[r * K + k, ids].cpu().numpy().astype(np.float64)
                del cache, p_out, s_out
        return results  # type: ignore[return-value]

    def hidden_states(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None,
                      token_ids: Optional[Sequence[Sequence[int]]] = None):
        """Last-position hidden states, float32 [N, len(layers), H], plus the label log-probs
        when `token_ids` is given (same forward). `layers` index the transformer's hidden-state
        tuple: 0 is the embedding output, i the output of block i, and the last entry
        (num_hidden_layers) is after the final norm, i.e. exactly what the lm_head reads; a
        negative index counts from that end. This is the feature the closed-form heads in
        `anyjev.heads` are fit on."""
        import torch

        n_blocks = int(self.model.config.num_hidden_layers)
        layers = [n_blocks] if layers is None else [(n_blocks + 1 + i) if i < 0 else i for i in layers]
        n = len(prompts)
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        order = sorted(range(n), key=lambda i: lengths[i])
        feats = np.zeros((n, len(layers), int(self.model.config.hidden_size)), dtype=np.float32)
        lps: List[Optional[np.ndarray]] = [None] * n
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt",
                                 padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                try:
                    out = self.model(**enc, position_ids=pos, logits_to_keep=1, output_hidden_states=True)
                except TypeError:
                    out = self.model(**enc, position_ids=pos, output_hidden_states=True)
                for li, layer in enumerate(layers):           # left padding: the last column is the last token
                    feats[idx, li] = out.hidden_states[layer][:, -1, :].float().cpu().numpy()
                if token_ids is not None:
                    lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                    for row, i in enumerate(idx):
                        ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                        lps[i] = lp[row, ids].cpu().numpy().astype(np.float64)
            del out
        return feats, lps

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
                logits = self._last_logits(enc, pos)
            lp = torch.log_softmax(logits, dim=-1)
            for row, i in enumerate(idx):
                ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                out[i] = lp[row, ids].cpu().numpy().astype(np.float64)
        return out  # type: ignore[return-value]
