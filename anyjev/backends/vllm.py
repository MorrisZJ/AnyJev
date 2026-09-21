"""vLLM backend over the OpenAI-compatible server.

One request per prompt with `max_tokens=1`, `allowed_token_ids` restricted to
the label tokens and `logprobs=K`. vLLM reports logprobs after its logit
processors, so the K entries are exactly the labels, normalized over them.
Prefix caching on the server makes the K permutations of one state cheap.

    vllm serve Qwen/Qwen3-8B --enable-prefix-caching
    Decider(VLLMBackend("http://localhost:8000", "Qwen/Qwen3-8B"))
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import urllib.request
from typing import List, Sequence

import numpy as np


class VLLMBackend:
    def __init__(self, base_url: str, model: str, tokenizer_name: str | None = None,
                 api_key: str = "EMPTY", workers: int = 16, timeout: float = 120.0):
        from transformers import AutoTokenizer

        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model)

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        body = {
            "model": self.name, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
            "logprobs": len(ids), "allowed_token_ids": list(ids),
        }
        req = urllib.request.Request(
            self.base_url + "/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]   # {token_str: logprob}
        # map back by token id: decode each id the same way the server renders it
        by_id = {}
        for tid in ids:
            tok = self.tokenizer.decode([tid])
            conv = self.tokenizer.convert_ids_to_tokens(tid)
            for key in (tok, conv):
                if key in top:
                    by_id[tid] = float(top[key])
                    break
        return np.array([by_id.get(tid, -30.0) for tid in ids], dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        with cf.ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, prompts, token_ids))
