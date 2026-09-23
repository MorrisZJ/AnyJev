"""Synthetic states for closed-form distillation: new cases in each typed-decisions workflow,
written by an open model from a few real examples, to be labelled by a teacher head.

    python -m bench.synth_states --model Qwen/Qwen3-8B --per-workflow 1200 --out bench/results_distill/synth

For every workflow the generator sees three real train cases (rendered JSON, different ones per
prompt) and writes one new case with the same schema and field types; the output is kept when
it parses as JSON with a top-level key set that occurs among the real cases and is not a duplicate. No labels are
produced here; the teacher head labels the states afterwards (`bench.distill_heads label`).
Sampling: temperature 1.0, top-p 0.95, thinking off. Output: <out>/<workflow>.json with the
states and the generation settings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

from anyjev.state import render_state
from bench.run import environment
from bench.tasks.typed_decisions import load

INSTRUCTION = (
    "You write realistic test cases for a {workflow} decision system. Below are {n} real cases, each a JSON "
    "object. Write ONE new case with exactly the same JSON schema (same keys, same nesting, same value types). "
    "Make it a different situation: different people, products, numbers, dates, wording and outcome. Cover "
    "the whole range of what happens in practice, including ambiguous or borderline situations. Output only "
    "the JSON object, nothing else.\n\n"
)


def build_prompt(tokenizer, workflow: str, exemplars: List[Dict[str, Any]]) -> str:
    body = INSTRUCTION.format(workflow=workflow.replace("_", " "), n=len(exemplars))
    for i, ex in enumerate(exemplars, 1):
        body += f"Case {i}:\n{render_state(ex)}\n\n"
    body += "New case:"
    messages = [{"role": "user", "content": body}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_case(text: str, key_sets: set) -> Dict[str, Any] | None:
    """The JSON object in a generation, or None (the reason is recorded by `reject_reason`)."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = max(parts, key=len).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or frozenset(obj.keys()) not in key_sets:
        return None
    return obj


def reject_reason(text: str, key_sets: set) -> str:
    t = text.strip()
    if t.count("{") > t.count("}"):
        return "truncated"
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return "no_json"
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return "invalid_json"
    if not isinstance(obj, dict):
        return "not_object"
    return "keys" if frozenset(obj.keys()) not in key_sets else "duplicate"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--per-workflow", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/results_distill/synth")
    ap.add_argument("--workflows", default="", help="comma list; default all four")
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    by_wf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_state = set()
    for d in load("train"):
        key = json.dumps(d.state, sort_keys=True)
        if key not in seen_state:
            seen_state.add(key)
            by_wf[d.workflow].append(d.state)
    workflows = args.workflows.split(",") if args.workflows else sorted(by_wf)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for wf in workflows:
        real = by_wf[wf]
        keys = {frozenset(s.keys()) for s in real}          # every schema that occurs in the real cases
        out: List[Dict[str, Any]] = []
        dupes = set(json.dumps(s, sort_keys=True) for s in real)
        n_gen = n_bad = 0
        reasons: Dict[str, int] = defaultdict(int)
        while len(out) < args.per_workflow:
            prompts = [build_prompt(tok, wf, rng.sample(real, 3)) for _ in range(args.batch_size)]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for t in texts:
                n_gen += 1
                obj = parse_case(t, keys)
                if obj is None:
                    n_bad += 1
                    reasons[reject_reason(t, keys)] += 1
                    continue
                key = json.dumps(obj, sort_keys=True)
                if key in dupes:
                    n_bad += 1
                    reasons["duplicate"] += 1
                    continue
                dupes.add(key)
                out.append(obj)
            print(f"   {wf:28s} kept {len(out):5d} / generated {n_gen:5d} (rejected {n_bad}: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ")", flush=True)
        out = out[:args.per_workflow]
        path = os.path.join(args.out, f"{wf}.json")
        with open(path, "w") as f:
            json.dump({"workflow": wf, "generator": args.model, "n": len(out), "generated": n_gen, "rejected": n_bad,
                       "reject_reasons": dict(reasons),
                       "settings": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                                    "max_new_tokens": args.max_new_tokens, "exemplars_per_prompt": 3},
                       "date": dt.datetime.now().isoformat(), "env": environment(), "states": out}, f)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
