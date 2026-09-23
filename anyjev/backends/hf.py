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
        self.n_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))

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
                      token_ids: Optional[Sequence[Sequence[int]]] = None,
                      positions: Optional[Sequence[Sequence[int]]] = None):
        """Hidden states from one forward per prompt batch.

        Returns (feats, lps, pos_feats): `feats` float32 [N, len(layers), H] at the LAST position;
        `lps` the label log-probs at that position when `token_ids` is given (same forward, else
        None entries); `pos_feats` float16 [N, P_max, len(layers), H] at the requested token
        indices per prompt (`positions[i]`, indices into the prompt's own tokens; zero-padded to
        the longest list) or None when `positions` is None. `layers` index the transformer's
        hidden-state tuple: 0 is the embedding output, i the output of block i, and the last entry
        (num_hidden_layers) is after the final norm, i.e. exactly what the lm_head reads; a
        negative index counts from that end. These are the features the closed-form heads in
        `anyjev.heads` are fit on."""
        import torch

        n_blocks = int(self.model.config.num_hidden_layers)
        layers = [n_blocks] if layers is None else [(n_blocks + 1 + i) if i < 0 else i for i in layers]
        n = len(prompts)
        H = int(self.model.config.hidden_size)
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        order = sorted(range(n), key=lambda i: lengths[i])
        feats = np.zeros((n, len(layers), H), dtype=np.float32)
        lps: List[Optional[np.ndarray]] = [None] * n
        pos_feats = None
        if positions is not None:
            p_max = max((len(p) for p in positions), default=0)
            pos_feats = np.zeros((n, p_max, len(layers), H), dtype=np.float16)
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
                if positions is not None:
                    T_pad = int(enc["input_ids"].shape[1])
                    n_tok = enc["attention_mask"].sum(dim=1).tolist()
                    for row, i in enumerate(idx):
                        if not positions[i]:
                            continue
                        if int(n_tok[row]) != lengths[i]:
                            raise RuntimeError(f"token count mismatch for prompt {i}: {n_tok[row]} vs {lengths[i]}")
                        cols = torch.as_tensor([T_pad - int(n_tok[row]) + int(p) for p in positions[i]],
                                               device=self.model.device)
                        for li, layer in enumerate(layers):
                            pos_feats[i, :len(positions[i]), li] = (
                                out.hidden_states[layer][row, cols, :].to(torch.float16).cpu().numpy())
                if token_ids is not None:
                    lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                    for row, i in enumerate(idx):
                        ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                        lps[i] = lp[row, ids].cpu().numpy().astype(np.float64)
            del out
        return feats, lps, pos_feats

    # ---- depth: a block loop that can stop early, capture every block, and resume ----------
    def _prepare(self, enc, pos):
        """Everything a transformers 4.5x decoder block needs besides the residual stream: the
        embeddings, the causal mask(s), cache positions and rotary embeddings. Mirrors
        `XModel.forward` so blocks can be run one at a time. Raises NotImplementedError for
        architectures whose forward differs (the caller falls back to output_hidden_states)."""
        import torch

        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
            raise NotImplementedError("no .model.layers / .model.embed_tokens on this architecture")
        try:
            from transformers import DynamicCache
            from transformers.masking_utils import create_causal_mask
        except ImportError as e:   # older transformers
            raise NotImplementedError(str(e))
        ids, mask = enc["input_ids"], enc["attention_mask"]
        embeds = inner.embed_tokens(ids)
        cache = DynamicCache()
        cache_position = torch.arange(0, embeds.shape[1], device=embeds.device)
        kw = dict(config=self.model.config, input_embeds=embeds, attention_mask=mask,
                  cache_position=cache_position, past_key_values=cache, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**kw)}
        types = {getattr(layer, "attention_type", "full_attention") for layer in inner.layers}
        if "sliding_attention" in types:
            from transformers.masking_utils import create_sliding_window_causal_mask
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kw)
        if not hasattr(inner, "rotary_emb"):
            raise NotImplementedError("no shared rotary embedding module on this architecture")
        rope = inner.rotary_emb(embeds, pos)
        scale = getattr(inner, "embed_scale", None)      # Gemma-style scaled embeddings
        if scale is not None:
            raise NotImplementedError("scaled embeddings (Gemma) are not supported by the block loop yet")
        return {"hidden": embeds, "masks": masks, "cache": cache, "cache_position": cache_position,
                "position_ids": pos, "rope": rope}

    def _run_layers(self, ctx, start: int, stop: int, capture=None):
        """Run blocks start..stop-1 (0-based) on ctx["hidden"], updating ctx["cache"]. `capture(i, h)`
        receives each block's output (1-based block index, matching the hidden-state tuple)."""
        inner = self.model.model
        h = ctx["hidden"]
        for i in range(start, stop):
            layer = inner.layers[i]
            out = layer(h, attention_mask=ctx["masks"][getattr(layer, "attention_type", "full_attention")],
                        position_ids=ctx["position_ids"], past_key_value=ctx["cache"],
                        cache_position=ctx["cache_position"], position_embeddings=ctx["rope"])
            h = out[0] if isinstance(out, tuple) else out
            if capture is not None:
                capture(i + 1, h)
        ctx["hidden"] = h
        ctx["layer"] = stop
        return h

    def hidden_states_to(self, prompts: Sequence[str], layers: Sequence[int], token_ids=None,
                         positions=None, max_layer: Optional[int] = None, lens_ids=None):
        """Like `hidden_states` but through the block loop: runs only the first `max_layer` blocks
        (default: the deepest requested layer), captures the requested layers without keeping every
        layer's full activations, and returns per-layer restricted logit-lens logits for `lens_ids`
        ([N, len(layers), len(lens_ids)]: final norm + the lm_head rows of those tokens at each
        captured layer). Layer index n_layers means 'after the final norm'. Returns
        (feats, lps, pos_feats, lens)."""
        import torch

        n_blocks = self.n_layers
        layers = [(n_blocks + 1 + i) if i < 0 else i for i in layers]
        deepest = max(layers)
        stop = min(n_blocks, max_layer if max_layer is not None else deepest)
        if deepest > stop and deepest != n_blocks:
            raise ValueError(f"requested layer {deepest} lies beyond max_layer {stop}")
        n = len(prompts)
        H = self.hidden_size
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        order = sorted(range(n), key=lambda i: lengths[i])
        feats = np.zeros((n, len(layers), H), dtype=np.float32)
        lps: List[Optional[np.ndarray]] = [None] * n
        pos_feats = None
        if positions is not None:
            p_max = max((len(p) for p in positions), default=0)
            pos_feats = np.zeros((n, p_max, len(layers), H), dtype=np.float16)
        lens = np.zeros((n, len(layers), len(lens_ids)), dtype=np.float32) if lens_ids is not None else None
        lens_t = torch.as_tensor(list(lens_ids), device=self.model.device) if lens_ids is not None else None
        inner = self.model.model
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            T_pad = int(enc["input_ids"].shape[1])
            n_tok = enc["attention_mask"].sum(dim=1).tolist()
            captured: Dict[int, "torch.Tensor"] = {}

            def grab(block_i, h, captured=captured):
                if block_i in layers:
                    captured[block_i] = h

            with torch.no_grad():
                ctx = self._prepare(enc, pos)

                if 0 in layers:
                    captured[0] = ctx["hidden"]
                self._run_layers(ctx, 0, stop, capture=grab)
                if n_blocks in layers or token_ids is not None:
                    final = inner.norm(ctx["hidden"])
                    if n_blocks in layers and stop == n_blocks:
                        captured[n_blocks] = final
                    elif n_blocks in layers:
                        captured[n_blocks] = final          # logit-lens style: norm of the truncated stream
                for li, layer in enumerate(layers):
                    h = captured[layer]
                    feats[idx, li] = h[:, -1, :].float().cpu().numpy()
                    if lens_t is not None:
                        hl = h[:, -1, :] if layer == n_blocks else inner.norm(h[:, -1, :])
                        lens[idx, li] = self._project(hl)[:, lens_t].float().cpu().numpy()
                    if positions is not None:
                        for row, i in enumerate(idx):
                            if not positions[i]:
                                continue
                            cols = torch.as_tensor([T_pad - int(n_tok[row]) + int(p) for p in positions[i]],
                                                   device=self.model.device)
                            pos_feats[i, :len(positions[i]), li] = h[row, cols, :].to(torch.float16).cpu().numpy()
                if token_ids is not None:
                    lp = torch.log_softmax(self._project(final[:, -1, :]), dim=-1)
                    for row, i in enumerate(idx):
                        ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                        lps[i] = lp[row, ids].cpu().numpy().astype(np.float64)
            del ctx
        return feats, lps, pos_feats, lens

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
