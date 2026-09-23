"""NanoJev's own "Untuned Qwen" maze baseline, run here for an apples-to-apples row.

Wraps scripts/evaluate_native_qwen_maze.py from the NanoJev checkout. The only
deviation: the predictor is built without `disable_native_triton`, which needs
torch 2.14's torch._native and does not exist on torch 2.5.

    python -m bench.providers.nanojev_native_maze --nanojev /path/to/NanoJev --episodes ... --out ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time


def summarize_edges(result):
    """Majority baseline and mean predicted p(clear) over every visited cell, added to the summary
    so the maze table can be regenerated from this file alone. Computed before observations are
    dropped from the artifact."""
    obs = [o for e in result["episodes"] for o in e.get("observations", [])]
    dirs = ("north", "east", "south", "west")
    clear = [bool(o["truth"][a]) for o in obs for a in dirs]
    pm = [o["probabilities"][a] for o in obs for a in dirs]
    if clear:
        result["summary"]["edge_majority"] = max(sum(clear), len(clear) - sum(clear)) / len(clear)
        result["summary"]["mean_p_true"] = sum(pm) / len(pm)
        result["summary"]["edge_questions"] = len(clear)


def checkout_info(nanojev_dir, episodes):
    """The NanoJev commit and the episodes file hash, so the comparison is pinned."""
    import hashlib
    import subprocess
    info = {"nanojev_dir": os.path.basename(os.path.abspath(nanojev_dir))}
    try:
        info["nanojev_commit"] = subprocess.run(["git", "-C", nanojev_dir, "rev-parse", "HEAD"], capture_output=True,
                                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["nanojev_commit"] = None
    with open(episodes, "rb") as f:
        info["episodes_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanojev", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-states", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", default="bench/results_nanojev")
    ap.add_argument("--keep-observations", action="store_true",
                    help="keep every per-cell observation in the JSON (~10 MB per run); off by default")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(args.nanojev, "scripts"))
    import evaluate_model_edges_maze as core  # noqa: E402
    import evaluate_native_qwen_navigation as nav  # noqa: E402
    from evaluate_native_qwen_maze import NativeBooleanPredictor  # noqa: E402

    class Predictor(nav.NativeQwenPredictor):
        def __init__(self, tokenizer, snapshot, token_ids, max_length, precision):
            import torch
            from transformers import AutoModelForCausalLM
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            self.torch, self.tokenizer, self.token_ids = torch, tokenizer, token_ids
            self.max_length, self.precision, self.calls = max_length, precision, 0
            self.model = AutoModelForCausalLM.from_pretrained(
                snapshot, local_files_only=True, trust_remote_code=False,
                attn_implementation="sdpa", torch_dtype=torch.float32).to("cuda:0").eval()
            self.model.config.use_cache = False
            self.parameter_count = sum(p.numel() for p in self.model.parameters())

    splits = {s.strip() for s in args.splits.split(",")}
    episodes = [json.loads(line) for line in open(args.episodes) if line.strip()]
    selected = [e for e in episodes if e["game"] == "scaled_maze" and e["split"] in splits]
    print(f"{len(selected)} maze episodes", flush=True)
    tokenizer, snapshot, token_ids = nav.load_tokenizer()
    engine = NativeBooleanPredictor(Predictor(tokenizer, snapshot, token_ids, args.max_length, "bf16"))
    t0 = time.time()
    result = core.run_exploration(selected, engine, args.window_size, args.max_steps, args.batch_states, 0)
    result.update(model=nav.MODEL, revision=nav.REVISION, level="native_AB_readout", engine="nanojev_native",
                  elapsed_seconds=time.time() - t0, date=dt.datetime.now().isoformat())
    summarize_edges(result)
    result.update(checkout_info(args.nanojev, args.episodes))
    for ep in result["episodes"]:
        ep.pop("steps", None)
        if not args.keep_observations:
            ep.pop("observations", None)
            ep.pop("verified_edges", None)
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "nanojev_native.Qwen3-0.6B.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    per = [(e["id"], e["status"], e["attempts"], e["collisions"]) for e in result["episodes"]]
    print(json.dumps({"summary": result["summary"], "episodes": per}, indent=1, default=float))


if __name__ == "__main__":
    main()
