"""One command: turn a model into a Jev, serve it, and measure what you got.

    python -m anyjev.pipeline Qwen/Qwen2.5-7B-Instruct --labels-from banking20

It truncates the model, starts vLLM on it, fits a head from your labels, measures accuracy and
milliseconds per decision on held-out states, tears the server down and prints a table. It runs
once per depth, so the table always has something to compare against and the truncation's cost
or gain is visible rather than asserted.

    model                     depth   accuracy   ECE   single   agent
    Qwen2.5-7B-Instruct      28/28      0.830  0.041   9.2 ms  9.6 ms
    Qwen2.5-7B-Instruct      18/28      0.850  0.038   7.6 ms  7.2 ms

Two knobs are on by default because measuring them is what this release is about:

**Depth.** A decision does not need the whole model: a closed-form head is flat from about two
thirds of the depth, and the blocks above that convert the answer into token space rather than
deciding anything. `anyjev.truncate` writes those blocks away into an ordinary smaller
checkpoint, so vLLM, transformers, a quantiser and a GGUF converter all take it unchanged. On
banking20 with Qwen2.5-7B at 18 of 28 blocks it is 1.2x faster **and two points more accurate**,
because a middle block is a better feature space for a linear head than the final one.

**Prefix caching.** AnyJev's shape is one state and several questions, so the state's keys and
values are computed once and every later question reads them. On the same setup that is 1.5x on
the multi-question workload and, as it should be, nothing at all on the one-question workload.

The `agent` column is the multi-question shape; `single` is one question per state. Both are
measured against a server this script started, so the numbers are the ones your deployment
gets, not a local forward loop's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

POOLER = '{"pooling_type":"LAST","normalize":false,"softmax":false}'


# ---------------------------------------------------------------- the server
class Served:
    """A vLLM pooling server for the life of a `with` block."""

    def __init__(self, model: str, port: int = 8000, gpu_fraction: float = 0.85,
                 prefix_caching: bool = True, quantization: Optional[str] = None,
                 extra: Sequence[str] = (), log: Optional[str] = None, timeout_s: float = 900.0):
        self.model, self.port, self.timeout_s = model, port, timeout_s
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model, "--served-model-name", "anyjev",
                    "--port", str(port), "--task", "embed",
                    "--gpu-memory-utilization", str(gpu_fraction),
                    "--override-pooler-config", POOLER]
        self.cmd += ["--enable-prefix-caching"] if prefix_caching else ["--no-enable-prefix-caching"]
        if quantization:
            self.cmd += ["--quantization", quantization]
        self.cmd += list(extra)
        self.log = log
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "Served":
        out = open(self.log, "w") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}"
                                   + (f"; see {self.log}" if self.log else ""))
            try:
                urllib.request.urlopen(self.url + "/v1/models", timeout=3).read()
                return self
            except (urllib.error.URLError, OSError):
                time.sleep(3)
        self.__exit__(None, None, None)
        raise TimeoutError(f"vLLM did not come up within {self.timeout_s:.0f}s")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)


# ---------------------------------------------------------------- one configuration
def measure(url: str, source: str, question, cal_x, cal_y, te_x, te_y,
            agent_questions: Sequence[Any] = (), agent_states: int = 64,
            repeats: int = 3) -> Dict[str, Any]:
    """Fit a head through the server, then time and score it on held-out states.

    Timings are the median of `repeats` passes. A single pass is not enough: the first version
    of this ran each workload once and reported the truncated model as both 1.33x faster and
    0.76x slower than the full one on the same workload, on two runs that differed only in
    noise."""
    from anyjev import Decider
    from anyjev.backends.vllm import VLLMBackend
    from bench.metrics import ece

    dec = Decider(VLLMBackend(url, "anyjev", tokenizer_name=source), level="L2")
    t0 = time.perf_counter()
    art = dec.fit_head(question, list(cal_x), list(cal_y), layers=[-1])
    fit_s = time.perf_counter() - t0

    single = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        P = np.stack([d.probs for d in dec.decide_batch(list(te_x), question, level="L2")])
        single.append(time.perf_counter() - t0)
    single_s = float(np.median(single))
    y = np.asarray(te_y)

    agent_ms = float("nan")
    if agent_questions:
        sub = list(te_x)[:agent_states]
        for eq in agent_questions:
            dec.fit_head(eq, list(cal_x)[:64], [i % eq.k for i in range(64)], layers=[-1])
        passes = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for eq in [question, *agent_questions]:
                dec.decide_batch(sub, eq, level="L2")
            passes.append(time.perf_counter() - t0)
        agent_ms = float(np.median(passes)) / (len(sub) * (1 + len(agent_questions))) * 1000
        agent_spread = (max(passes) - min(passes)) / float(np.median(passes))

    return {"accuracy": float((P.argmax(1) == y).mean()), "ece": float(ece(P, y)),
            "single_ms": single_s / len(te_x) * 1000, "agent_ms": agent_ms,
            "single_spread": (max(single) - min(single)) / single_s,
            "agent_spread": locals().get("agent_spread", float("nan")),
            "repeats": repeats, "fit_s": fit_s, "head": art["method"],
            "n_test": len(te_x), "n_calib": len(cal_x)}


# ---------------------------------------------------------------- the command
def main(argv=None):
    ap = argparse.ArgumentParser(description="convert, serve, measure, print a table")
    ap.add_argument("model")
    ap.add_argument("--labels-from", default="banking20",
                    help="a registered bench task to take labelled states from")
    ap.add_argument("--depths", default="",
                    help="comma-separated block counts; default: the full model, and 2/3 of it")
    ap.add_argument("--only-full", action="store_true",
                    help="just the full model; by default both depths are measured so the "
                         "table has something to compare against")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--calib", type=int, default=300)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--gpu-fraction", type=float, default=0.85)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--quantization", default=None, help="e.g. fp8; costs accuracy, see the docs")
    ap.add_argument("--keep-truncated", default="", help="where to keep truncated checkpoints")
    ap.add_argument("--agent-questions", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3, help="timing passes; the median is reported")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from transformers import AutoConfig

    from anyjev.question import Question
    from anyjev.truncate import truncate
    from bench.tasks import get_task

    task = get_task(args.labels_from)
    test, calib = task.split(args.n_test, args.calib, seed=0)
    q = task.question
    total = int(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    depths = [int(x) for x in args.depths.split(",") if x]
    if not depths:
        depths = [total] if args.only_full else [total, max(1, round(total * 2 / 3))]
    extra_q = [Question.choice(t, o, name=n) for n, t, o in (
        ("urgency", "How urgent is this message?", ["can wait", "this week", "today", "now"]),
        ("sentiment", "What is the tone?", ["calm", "annoyed", "angry"]),
        ("handoff", "Does this need a human?", ["no", "yes"]),
        ("refund", "Is money back being asked for?", ["no", "yes"]),
    )][:args.agent_questions]

    keep = args.keep_truncated or tempfile.mkdtemp(prefix="anyjev-trunc-")
    rows: List[Dict[str, Any]] = []
    try:
        for blocks in depths:
            if blocks >= total:
                served_model, label = args.model, f"{total}/{total}"
            else:
                served_model = os.path.join(keep, f"{os.path.basename(args.model)}-b{blocks}")
                if not os.path.exists(served_model):
                    truncate(args.model, blocks, served_model)
                label = f"{blocks}/{total}"
            print(f"\n[anyjev] serving {label} ...", flush=True)
            with Served(served_model, port=args.port, gpu_fraction=args.gpu_fraction,
                        prefix_caching=not args.no_prefix_caching,
                        quantization=args.quantization,
                        log=os.path.join(keep, f"vllm-b{blocks}.log")) as srv:
                r = measure(srv.url, args.model, q, [s for s, _ in calib], [y for _, y in calib],
                            [s for s, _ in test], [y for _, y in test], extra_q,
                            repeats=args.repeats)
            r.update({"model": args.model, "blocks": blocks, "total_blocks": total,
                      "depth": label, "prefix_caching": not args.no_prefix_caching,
                      "quantization": args.quantization})
            rows.append(r)
            print(f"[anyjev] {label}: accuracy {r['accuracy']:.3f}  ECE {r['ece']:.3f}  "
                  f"single {r['single_ms']:.1f} ms  agent {r['agent_ms']:.1f} ms", flush=True)
    finally:
        if not args.keep_truncated:
            shutil.rmtree(keep, ignore_errors=True)

    name = args.model.split("/")[-1]
    print(f"\n{'model':<26}{'depth':>8}{'accuracy':>10}{'ECE':>7}{'single':>9}{'agent':>9}")
    for r in rows:
        print(f"{name:<26}{r['depth']:>8}{r['accuracy']:>10.3f}{r['ece']:>7.3f}"
              f"{r['single_ms']:>7.1f} ms{r['agent_ms']:>7.1f} ms"
              f"   (spread {r['single_spread']:.0%} / {r['agent_spread']:.0%})")
    if len(rows) > 1:
        a, b = rows[0], rows[-1]
        print(f"\n  {b['depth']} against {a['depth']}: "
              f"{b['accuracy'] - a['accuracy']:+.3f} accuracy, "
              f"{a['single_ms'] / b['single_ms']:.2f}x single, "
              f"{a['agent_ms'] / b['agent_ms']:.2f}x agent "
              f"({1 + len(extra_q)} questions per state)")

    out = {"model": args.model, "task": task.name, "rows": rows,
           "date": dt.date.today().isoformat()}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nwrote {args.out}")
    return out


if __name__ == "__main__":
    main()
