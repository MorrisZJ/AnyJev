"""The packaged Jev-mode (L2) demo: `python -m demo.jev_mode`.

    python -m demo.jev_mode --backend fake       # synthetic model, numpy only, seconds, no download
    python -m demo.jev_mode                      # Qwen3-4B (GPU) or 1.7B (CPU) + the shipped heads on typed-decisions
    python -m demo.jev_mode --state-file ticket.json --question customer_service.category   # one state
    python -m demo.jev_mode --state-file ticket.json --question @q.json --labels-file labels.json

Sections: [1/4] the levels on held-out states (L0 zero-label vs L2 head; raw too on the synthetic model),
[2/4] the same question reworded and re-listed with no new labels (routing, label-free recentring, reversed
options), [3/4] a new head from a few labels, [4/4] export -> reload round trip. Every number printed is
measured at run time; `--json PATH` writes them all with the environment. The synthetic model's numbers are
planted and say so on screen; real numbers with a JSON source are in docs/jev_mode.md. Public API only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    raise SystemExit("AnyJev demo: numpy is required. Install with: pip install numpy") from None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anyjev import Decider, Question  # noqa: E402
from anyjev.readout import build_prompt, render_chat_parts  # noqa: E402
from anyjev.state import render_state  # noqa: E402
from bench import metrics  # noqa: E402
from bench.run import environment  # noqa: E402
from demo.games.common import bar, fmt_table, paint, reversed_question, use_color, write_json  # noqa: E402

# numpy on macOS Accelerate reports spurious "... encountered in matmul" for finite float64 products; silence those
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

# ---------------------------------------------------------------- constants
_SPEC = """agent_trace_observability: action=choice4 needs_review=noul2 outcome=choice4 risk=score4 urgency=score4
customer_service: action=choice5 category=choice5 churn_risk=score4 needs_human=noul2 urgency=score4
invoice_processing: discrepancy_severity=score4 disposition=choice4 duplicate=noul2 matches_order=noul2 urgency=score4
security_incidents: credential_compromise=noul2 disposition=choice4 severity=score5 true_positive=noul2 urgency=score4
"""
# the 20 LocalLLaMA/typed-decisions questions in dataset order: workflow.qname -> (kind, K)
TYPED_QUESTIONS: Dict[str, Tuple[str, int]] = {f"{w}.{n}": (k[:-1], int(k[-1]))
                                              for w, rest in (ln.split(": ") for ln in _SPEC.split("\n")[:-1])
                                              for n, k in (x.split("=") for x in rest.split())}
TYPED_QUESTION_NAMES: List[str] = list(TYPED_QUESTIONS)
WORKFLOWS = [ln.split(":")[0] for ln in _SPEC.split("\n")[:-1]]
WORKFLOW_HINTS = {"agent_trace_observability": ("trace", "agent"), "customer_service": ("conversation", "customer"),
                  "invoice_processing": ("invoice",), "security_incidents": ("alert", "security", "incident")}
SHIPPED_MODELS = "Qwen3-1.7B, 4B, 8B, 30B-A3B-Instruct-2507, 32B"
PARAPHRASES = os.path.join(ROOT, "bench", "tasks", "typed_paraphrases.json")
REWORD_VARIANTS = ("w1", "w2", "w3", "o1")
DIAG_KEYS = ("readout", "blocks_executed", "n_blocks", "early_stop", "temperature", "n_calib", "listing", "routed_from",
             "reordered", "adapted", "adapt_n", "permutations", "prior_method", "order_flip_raw", "order_flip_l0",
             "raw_probs", "answer_mass")
# the synthetic world: tuned so that raw < L0 < L2 and a rewording costs the head as is more than recentring recovers
HANDLERS = ["billing", "technical", "sales", "other"]
TICKETS = {"billing": ["I was charged twice for last month", "the invoice total does not match my plan",
                       "my refund has not arrived after two weeks", "the card on file was declined twice",
                       "why did the price go up on my renewal"],
           "technical": ["the app crashes every time I open the receipts page", "login fails with a 500 error",
                         "exports hang at 99 percent and never finish", "push notifications stopped on Android",
                         "the dashboard shows stale data after a sync"],
           "sales": ["we need a quote for 200 seats with SSO", "is there a discount for a three-year contract",
                     "can we add the analytics add-on to our plan", "we want a demo of the enterprise tier",
                     "what is the pricing for the education programme"],
           "other": ["please delete my account and all my data", "how do I change the language of the interface",
                     "where can I find your accessibility statement", "I want to unsubscribe from the newsletter",
                     "the office address on your website is out of date"]}
FAKE = {"position_bias": [3.0, 0.0, 0.0, 0.0], "temperature": 0.7, "logit_noise": 3.0, "wording_shift": 3.0,
        "layer_noise": 0.1, "block": 3, "n_tickets": 200, "n_labelled": 150, "budgets": (16, 50, 150)}
FAKE_REWORDING = "Who should take this ticket?"
UNRELATED = Question.choice("Is this state about billing?", ["yes", "no", "unclear"], name="unrelated")


def fail(msg: str) -> None:
    raise SystemExit(msg)


def short(path: str) -> str:
    return os.path.relpath(path, ROOT) if os.path.abspath(path).startswith(ROOT + os.sep) else path


def say(text: str = "", bold: bool = False) -> None:
    print(paint(text, bold=True) if bold and use_color() else text, flush=True)


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_duration(s: float) -> str:
    return f"{s:.1f} s" if s < 90 else (f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h")


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    return fn(*a, **kw), time.perf_counter() - t0


def pick(d) -> Dict[str, Any]:
    return {k: d.diagnostics[k] for k in DIAG_KEYS if k in d.diagnostics}


# ---------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m demo.jev_mode", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("mode")
    g.add_argument("--backend", default="hf", choices=["hf", "fake"], help="hf: a transformers model (default); "
                   "fake: the synthetic model, no download, seconds")
    g.add_argument("--state-file", help="your own state (.json rendered by anyjev.state.render_state, else text); "
                   "switches to single-state mode")
    g.add_argument("--question", help="with --state-file: a shipped name (customer_service.category), inline JSON "
                   "or @file.json; otherwise an alias of --questions")
    g.add_argument("--labels-file", help='with --state-file and an own question: JSON list of '
                   '{"state": ..., "label": <option index>}; fits a head first')
    g.add_argument("--save-head", help="write export_artifacts() of the freshly fit head(s)")
    g.add_argument("--lifecycle", action="store_true", help="the deployment lifecycle on one question: day 0 at L0, "
                   "labels arriving one at a time (Decider.observe) until the head solves itself, a rewording served "
                   "and recentred from traffic, then export / restart / load")
    g.add_argument("--stream", type=int, default=150, help="--lifecycle: labelled requests in the stream")
    g.add_argument("--fit-at", type=int, default=30, help="--lifecycle: observe() solves the head at this count")
    g = ap.add_argument_group("model (hf)")
    g.add_argument("--model", help="the Hub id the heads file was built on; default Qwen/Qwen3-4B on a GPU, "
                   "Qwen/Qwen3-1.7B on a CPU")
    g.add_argument("--heads", help="heads file; default anyjev-heads/<model>.json; none = fit-only run")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    g.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                   help="auto = bfloat16 on cuda/mps, float32 on cpu")
    g.add_argument("--batch-size", type=int, help="default 16 on cuda, 4 on cpu/mps")
    g = ap.add_argument_group("data (hf, typed-decisions)")
    g.add_argument("--questions", help="all | <workflow> | <workflow>.<qname>[,...]; default all on cuda, "
                   "invoice_processing.disposition on cpu/mps")
    g.add_argument("--n-test", type=int, help="held-out test states per question (first N in dataset order); "
                   "default 100 on cuda, 20 on cpu/mps; min 8")
    g.add_argument("--reword", default="w2", help="rewording for [2/4]: w1, w2, w3, o1, none, or free text "
                   "(free text needs exactly one question)")
    g.add_argument("--adapt-after", type=int, default=30, help="Decider(adapt_min_n=N); clipped to --n-test")
    g.add_argument("--fit-labels", type=int, default=50, help="labels from the train split for [3/4]; 0 skips it")
    g.add_argument("--block", type=int, help="block the new head reads; default the heads file's block, else the "
                   "library's candidate blocks")
    g = ap.add_argument_group("output")
    g.add_argument("--json", help="write every number printed, plus the environment")
    g.add_argument("--no-color", action="store_true")
    g.add_argument("--seed", type=int, default=0, help="fake backend: synthetic tickets and the label subsets")
    args = ap.parse_args(argv)
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    if args.question and not args.state_file and not args.questions:
        args.questions = args.question
    return args


# ---------------------------------------------------------------- questions
def resolve_question_names(spec: str) -> List[str]:
    """`all`, a workflow, or workflow.qname[,...] (a `typed.` prefix is accepted) -> names in dataset order."""
    out: List[str] = []
    for tok in spec.split(","):
        t = tok.strip()
        t = t[len("typed."):] if t.startswith("typed.") else t
        if t == "all":
            out += TYPED_QUESTION_NAMES
        elif t in WORKFLOWS:
            out += [n for n in TYPED_QUESTION_NAMES if n.startswith(t + ".")]
        elif t in TYPED_QUESTIONS:
            out.append(t)
        else:
            groups = [(w, [n.split(".", 1)[1] for n in TYPED_QUESTION_NAMES if n.startswith(w + ".")])
                      for w in WORKFLOWS]
            choices = ", ".join(f"{w}.{{{','.join(qs)}}}" for w, qs in groups)
            fail(f"unknown question {t!r}; choose from: {choices}, a workflow name, or all")
    return [n for n in TYPED_QUESTION_NAMES if n in set(out)]


def build_question(kind: str, text: str, options=None, name=None, bins: int = 5, scale=(0.0, 1.0)) -> Question:
    if kind == "choice":
        return Question.choice(text, options, name=name)
    if kind == "noul":
        return Question.noul(text, name=name)
    if kind == "score":
        return Question.score(text, levels=options, name=name) if options else \
            Question.score(text, bins=bins, scale=tuple(scale), name=name)
    fail(f"unknown question kind {kind!r}; choose choice, noul or score")


def parse_question_spec(spec: str) -> Question:
    """Inline JSON or @file.json: {"kind":"choice","text":...,"options":[...]}, {"kind":"noul","text":...},
    {"kind":"score","text":...,"levels":[...]} or {"kind":"score","text":...,"bins":5}."""
    raw = open(spec[1:]).read() if spec.startswith("@") else spec
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        fail(f"--question is neither a shipped name nor JSON ({e}); expected a shipped name, inline JSON or @file.json")
    return build_question(d.get("kind", "choice"), d.get("text", ""), d.get("options") or d.get("levels"),
                          d.get("name", "own"), int(d.get("bins", 5)), d.get("scale", (0.0, 1.0)))


def question_from_head(head: Dict[str, Any]) -> Question:
    """Rebuild the question a shipped head was fit on from the artifact's kind / text / options."""
    return build_question(head.get("kind"), head.get("text"), head.get("options"), head.get("question_id"))


_PARAPHRASES: Optional[Dict[str, Any]] = None


def reworded(q: Question, name: str, variant: str) -> Optional[Question]:
    """The same question under another wording: a shipped variant (w1/w2/w3 rewrite the text, o1 the option
    descriptions; bench/tasks/typed_paraphrases.json), free text, or None for `none`."""
    global _PARAPHRASES
    if variant == "none":
        return None
    if variant not in REWORD_VARIANTS:
        return dataclasses.replace(q, text=variant)
    if _PARAPHRASES is None:
        _PARAPHRASES = json.load(open(PARAPHRASES))
    spec = _PARAPHRASES.get("typed." + name, {}).get(variant) or {}
    kw: Dict[str, Any] = {k: v for k, v in (("text", spec.get("text")),
                                             ("options", tuple(spec["options"]) if spec.get("options") else None)) if v}
    return dataclasses.replace(q, **kw) if kw else None


# ---------------------------------------------------------------- heads
@dataclass
class HeadsInfo:
    path: str
    data: Dict[str, Any]
    block: int
    n_blocks: int
    n_heads: int
    built: str


def heads_info(args) -> Optional[HeadsInfo]:
    """The shipped heads file for the model, or None (with a warning) when there is none or `--heads none`."""
    if args.backend == "fake" or args.heads == "none":
        return None
    path = args.heads or os.path.join(ROOT, "anyjev-heads", args.model.replace("/", "__") + ".json")
    if not os.path.exists(path):
        say(f"warning: no shipped heads for {args.model} at {short(path)} (shipped: {SHIPPED_MODELS}); "
            "running fit-only: sections [1/4] and [2/4] use the head fit in [3/4]")
        return None
    d = json.load(open(path))
    blocks = [int(h["layer_abs"]) for h in d.get("heads", {}).values()]
    env = d.get("env", {})
    built = f"{str(d.get('date', ''))[:10]} with anyjev {d.get('anyjev', '?')}" + \
        (f" on {env['gpu']}" if env.get("gpu") else "")
    return HeadsInfo(path, d, max(set(blocks), key=blocks.count) if blocks else 0, int(d.get("n_blocks", 0)),
                     len(blocks), built)


def load_heads(dec: Decider, info: HeadsInfo) -> int:
    try:
        return dec.load_artifacts(info.data)
    except ValueError:
        fail(f"heads file was fit on {info.data.get('model')}, backend is {dec.backend.name}: "
             "pass the exact Hub id, or --heads none")


def shipped_question(info: HeadsInfo, name: str) -> Question:
    """workflow.qname -> the question rebuilt from the heads file. Five qnames are shared by several workflows and
    the file carries no workflow, so the question text is matched on a workflow keyword."""
    workflow, qname = name.split(".", 1)
    hits = [h for h in info.data["heads"].values() if h.get("question_id") == qname]
    if len(hits) > 1:
        hits = [h for h in hits if any(w in str(h.get("text", "")).lower() for w in WORKFLOW_HINTS[workflow])]
    if len(hits) != 1:
        fail(f"the heads file {info.path} has no head for {name}")
    return question_from_head(hits[0])


# ---------------------------------------------------------------- preflight and backend
def preflight(args) -> str:
    """Checks in order, stopping at the first failure; returns the mode: single, fake or typed."""
    mode = "single" if args.state_file else ("lifecycle" if args.lifecycle else
                                              ("fake" if args.backend == "fake" else "typed"))
    if mode == "single" and not args.question:
        fail("--state-file needs --question (a shipped name, inline JSON, or @file.json)")
    if mode == "typed" or (mode == "lifecycle" and args.backend == "hf"):
        try:
            import datasets  # noqa: F401
        except ImportError:
            fail("AnyJev demo: the typed-decisions run needs the 'datasets' package. Install with: pip install datasets"
                 '   (or: pip install "anyjev[bench]"). --backend fake and --state-file need no dataset.')
    if args.backend == "hf":
        try:
            import torch
            import transformers  # noqa: F401
        except ImportError:
            fail('AnyJev demo: the hf backend needs torch and transformers. Install with: pip install "anyjev[hf]"'
                 "   (or try the synthetic model first: python -m demo.jev_mode --backend fake)")
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.dtype == "auto":
            args.dtype = "float32" if args.device == "cpu" else "bfloat16"
        elif args.dtype == "bfloat16" and args.device == "cpu":
            say("warning: bfloat16 on CPU is slow without AMX/AVX512-BF16; --dtype float32 is the CPU default")
        gpu = args.device == "cuda"
        args.model = args.model or ("Qwen/Qwen3-4B" if gpu else "Qwen/Qwen3-1.7B")
        args.batch_size = args.batch_size or (16 if gpu else 4)
        args.questions = args.questions or ("all" if gpu else "invoice_processing.disposition")
        args.n_test = args.n_test or (100 if gpu else 20)
    if mode == "lifecycle" and args.backend == "hf":
        if not args.questions or args.questions == "all":
            args.questions = "invoice_processing.discrepancy_severity"
        names = resolve_question_names(args.questions)
        if len(names) != 1:
            fail("--lifecycle runs one question (--questions workflow.qname)")
        if args.stream < max(args.fit_at, 8, 2 * TYPED_QUESTIONS[names[0]][1]):
            first = max(args.fit_at, 8, 2 * TYPED_QUESTIONS[names[0]][1])
            fail(f"--stream {args.stream} is below the first fit at {first}")
    if mode == "typed":
        args.heads_info = heads_info(args)
        if args.heads_info is None and args.fit_labels == 0:
            fail("nothing to run: no heads file and --fit-labels 0")
        names = resolve_question_names(args.questions)
        if args.n_test < 8:
            fail(f"--n-test {args.n_test} is below the minimum of 8")
        for n in names:
            need = max(8, 2 * TYPED_QUESTIONS[n][1])
            if 0 < args.fit_labels < need:
                fail(f"--fit-labels {args.fit_labels} is below max(8, 2K) = {need} for {n} (K={TYPED_QUESTIONS[n][1]})")
        if args.reword not in REWORD_VARIANTS + ("none",) and len(names) != 1:
            fail("a custom rewording needs exactly one question (--questions workflow.qname)")
        if args.adapt_after > args.n_test:
            say(f"--adapt-after {args.adapt_after} clipped to --n-test {args.n_test}")
            args.adapt_after = args.n_test
    return mode


def make_backend(args):
    if args.backend == "fake":
        return fake_world(args.seed)[0]
    from anyjev.backends.hf import HFBackend
    try:
        return HFBackend(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    except OSError as e:
        fail(f"{e}\ncheck the model id, your network, HF_TOKEN for gated models, or HF_HUB_OFFLINE=1 with a warm cache")


def warn_early_stop(dec: Decider, done: List[bool]) -> None:
    if dec.early_stop_error and not done:
        done.append(True)
        import transformers
        say(f"warning: early stop unavailable on this transformers version ({transformers.__version__}); hidden states "
            "read through a full forward, blocks_executed reports the head's block but the cost is the plain forward; "
            f"pip install -U transformers   [{dec.early_stop_error}]")


# ---------------------------------------------------------------- evaluation
@dataclass
class Row:
    n: int
    acc: float
    ece: Optional[float]
    conf: np.ndarray
    correct: np.ndarray
    answers: List[str]
    probs: np.ndarray            # in the decisions' own option order
    ms: float
    prompts: int
    blocks: int
    n_blocks: int
    readout: str
    diag: Dict[str, Any]

    def flip_vs(self, other: "Row") -> float:
        return float(np.mean([a != b for a, b in zip(self.answers, other.answers)]))

    def to_dict(self) -> Dict[str, Any]:
        return {"acc": self.acc, "ece": self.ece, "n": self.n, "prompts_per_decision": self.prompts,
                "blocks_executed": self.blocks, "n_blocks": self.n_blocks, "ms_per_decision": self.ms,
                "readout": self.readout}


def evaluate(decs: Sequence[Any], gold: Sequence[str], canonical: Sequence[str], seconds: float, n_blocks: int) -> Row:
    """Accuracy / ECE against gold option texts, probabilities mapped to the canonical option order (a reversed
    listing is scored like the original). ECE: 15 equal-mass bins, only for n >= 50."""
    idx = {o: i for i, o in enumerate(canonical)}
    P = np.zeros((len(decs), len(canonical)))
    for i, d in enumerate(decs):
        for o, p in d.distribution.items():
            P[i, idx[o]] = p
    y = np.asarray([idx[g] for g in gold])
    correct = (P.argmax(1) == y).astype(float)
    diag = decs[-1].diagnostics
    return Row(n=len(decs), acc=float(correct.mean()), ece=metrics.ece(P, y) if len(decs) >= 50 else None,
               conf=P.max(1), correct=correct, answers=[d.argmax for d in decs],
               probs=np.stack([d.probs for d in decs]), ms=1000.0 * seconds / len(decs),
               prompts=int(diag.get("permutations", 1)), blocks=int(diag.get("blocks_executed", n_blocks)),
               n_blocks=int(diag.get("n_blocks", n_blocks)), readout=str(diag.get("readout", decs[-1].level)),
               diag=pick(decs[-1]))


def pooled(rows: Sequence[Optional[Row]]) -> Dict[str, Any]:
    """Accuracy and ECE (15 equal-mass bins on top-1 confidence) over the concatenated decisions."""
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"n": 0, "acc": None, "ece": None}
    conf, correct = np.concatenate([r.conf for r in rows]), np.concatenate([r.correct for r in rows])
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    chunks = [c for c in np.array_split(np.arange(len(conf)), min(15, len(conf))) if len(c)]
    ece = sum(len(c) / len(conf) * abs(conf[c].mean() - correct[c].mean()) for c in chunks)
    return {"n": int(len(conf)), "acc": float(correct.mean()), "ece": float(ece)}


def fit_line(art: Dict[str, Any], seconds: float) -> str:
    p = art.get("params", {})
    hp = f"shrinkage {p['shrinkage']}" if "shrinkage" in p else (f"lambda {p['lam']}" if "lam" in p else "")
    return (f"fit_head: {art['method']} ({hp}), listing {p.get('listing', 'canonical')}, block {art['layer_abs']}, "
            f"out-of-fold acc {art['cv']['oof_acc']:.2f}, temperature {art['temperature']:.2f}, {seconds:.2f} s")


def routing_rows(be, serve: Dict[str, Any], q: Question, q_rew: Optional[Question], states, gold, adapt_after: int,
                 n_blocks: int) -> Dict[str, Row]:
    """as_is / recentred / l0_reworded (when the rewording routes) and reversed (choice only). Fresh deciders per
    row so no row's running statistics leak into another (as bench.paraphrase_study does)."""
    out: Dict[str, Row] = {}

    def served(dec: Decider, qq: Question, level: str, tag: str) -> None:
        if level == "L0" or dec.route(qq) is not None:
            if tag == "recentred":
                dec.decide_batch(states, qq, level=level)     # first pass: the requests only feed the statistics
            decs, s = timed(dec.decide_batch, states, qq, level=level)
            out[tag] = evaluate(decs, gold, q.options, s, n_blocks)

    if q_rew is not None:
        for tag, dec in (("as_is", Decider(be, adapt=False)),
                         ("recentred", Decider(be, adapt="routed", adapt_min_n=adapt_after))):
            dec.load_artifacts(serve)
            served(dec, q_rew, "L2", tag)
        served(Decider(be), q_rew, "L0", "l0_reworded")
    if q.kind == "choice":
        dec = Decider(be, adapt=False)
        dec.load_artifacts(serve)
        served(dec, reversed_question(q), "L2", "reversed")
    return out


def routing_json(rows: Dict[str, Row], exact: Row, q_rew: Optional[Question]) -> Dict[str, Any]:
    def rec(tag: str, *extra: str) -> Optional[Dict[str, Any]]:
        r = rows.get(tag)
        if r is None:
            return None
        return {"acc": r.acc, "flip_vs_exact": r.flip_vs(exact), **{k: r.diag.get(k) for k in extra}}
    return {"reworded_text": q_rew.text if q_rew else None, "exact": exact.acc,
            "routed_from": (rows.get("as_is") or rows.get("reversed") or exact).diag.get("routed_from"),
            "as_is": rec("as_is", "routed_from"), "recentred": rec("recentred", "routed_from", "adapted", "adapt_n"),
            "reversed": rec("reversed", "reordered"),
            "l0_reworded": rows["l0_reworded"].acc if "l0_reworded" in rows else None}


def roundtrip(be, export: Dict[str, Any], items: Sequence[Tuple[Question, Sequence[Any], np.ndarray]]) -> Dict:
    """export -> a temporary heads.json -> a fresh Decider -> load_artifacts -> max |dp| against the decisions of the
    decider that fit (or loaded) the heads; the artifact stores float32. Also: level="auto" without a head -> L0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "heads.json")
        with open(path, "w") as f:
            json.dump(export, f)
        d = Decider(be)
        n = d.load_artifacts(path)
        dp = max(float(np.abs(np.stack([x.probs for x in d.decide_batch(s, q, level="L2")]) - p).max())
                 for q, s, p in items)
        # an option set no head can route (a noul question would route to any noul head: same Yes / No options)
        auto = d.decide(items[0][1][0], [UNRELATED], level="auto")["unrelated"].level
        return {"n_heads": n, "bytes": os.path.getsize(path), "max_abs_dp": dp,
                "n_decisions": sum(len(s) for _, s, _ in items), "auto_without_head": auto, "path": path, "decider": d}


def roundtrip_line(rt: Dict[str, Any], what: str) -> str:
    size = f"{rt['bytes'] / 1e6:.1f} MB" if rt["bytes"] >= 1e6 else f"{rt['bytes'] / 1024:.0f} KB"
    return (f"{what} -> {rt['path']} ({rt['n_heads']} head{'s' if rt['n_heads'] != 1 else ''}, {size}) -> new Decider "
            f"-> load_artifacts -> max |dp| over {rt['n_decisions']} decisions: {rt['max_abs_dp']:.1e} "
            "(float32 artifact)")


# ---------------------------------------------------------------- the synthetic walkthrough
def fake_world(seed: int = 0):
    """200 templated tickets over 4 handlers and a FakeBackend whose output layer sees the content through a
    position bias and per-position noise, whose hidden state carries a clean code of the right option, and whose
    hidden state moves by a per-feature affine map under a rewording (FakeBackend docstring)."""
    from anyjev.backends.fake import FakeBackend
    rng = np.random.RandomState(seed)
    classes = np.repeat(np.arange(len(HANDLERS)), FAKE["n_tickets"] // len(HANDLERS))
    rng.shuffle(classes)
    labels = [int(c) for c in classes]
    states = [f"Ticket {i}: {TICKETS[HANDLERS[c]][rng.randint(5)]}" for i, c in enumerate(labels)]
    truth = {s: HANDLERS[y] for s, y in zip(states, labels)}
    be = FakeBackend(lambda s, o: 3.0 if truth.get(s) == o else 0.0, position_bias=FAKE["position_bias"],
                     temperature=FAKE["temperature"], logit_noise=FAKE["logit_noise"],
                     wording_shift=FAKE["wording_shift"])
    be.layer_noise = FAKE["layer_noise"]        # keeps the block-3 code readable (the default 0.4 buries it)
    return be, Question.choice("Which handler should take this ticket?", HANDLERS, name="route"), states, labels


def run_fake(args) -> Dict[str, Any]:
    t_run = time.perf_counter()
    be, q, states, labels = fake_world(args.seed)
    n_lab = FAKE["n_labelled"]
    tr_s, tr_y, te_s, te_y = states[:n_lab], labels[:n_lab], states[n_lab:], labels[n_lab:]
    gold = [HANDLERS[y] for y in te_y]
    nb, block = be.n_layers, FAKE["block"]
    say(f"AnyJev Jev-mode demo  |  backend fake (synthetic model, {nb} blocks, hidden {be.hidden_size})  |  "
        "no download", bold=True)
    bias = ",".join(f"{b:g}" for b in FAKE["position_bias"])
    say(f"Planted: output layer = content + position bias [{bias}] + noise; hidden state = clean code of the right "
        "option; rewording = shift + rescale of that state.")
    say(f"{len(states)} tickets, {len(HANDLERS)} handlers. {n_lab} labelled -> head. {len(te_s)} held out -> every "
        "number below.\n")
    say(f"[1/4] Levels on the {len(te_s)} held-out tickets", bold=True)
    dec, d0 = Decider(be), Decider(be)
    art, t_fit = timed(dec.fit_head, q, tr_s, tr_y, layers=[block])
    rows: Dict[str, Row] = {}
    for level, dd in (("raw", d0), ("L0", d0), ("L2", dec)):
        decs, s = timed(dd.decide_batch, te_s, q, level=level)
        rows[level] = evaluate(decs, gold, q.options, s, nb)
    say(fmt_table([[lv, n_lab if lv == "L2" else 0, r.prompts, f"{r.blocks} of {r.n_blocks}", fmt(r.acc, 2),
                    fmt(r.ece, 2)] for lv, r in rows.items()],
                  ["level", "labels", "prompts per decision", "blocks executed", "acc", "ece"]))
    say(fit_line(art, t_fit))
    say("(synthetic numbers: the separation is planted. Real numbers: --backend hf, or docs/jev_mode.md)\n")
    say("[2/4] The same question, reworded and re-listed (no new labels)", bold=True)
    export = dec.export_artifacts()
    q_rew = Question.choice(FAKE_REWORDING, HANDLERS, name="route_reworded")
    rr = routing_rows(be, export, q, q_rew, te_s, gold, args.adapt_after, nb)
    ro = routing_json(rr, rows["L2"], q_rew)
    table = [["original wording (exact head)", "-", "no", fmt(rows["L2"].acc, 2), "-"]]
    for tag, label, adapted in (("as_is", f'"{FAKE_REWORDING}", head as is', "no"),
                                ("recentred", f'"{FAKE_REWORDING}", recentred on {{n}} reqs', "yes ({n})"),
                                ("reversed", "options listed in reverse, probabilities remapped", "no")):
        if tag in rr:
            n_ad = rr[tag].diag.get("adapt_n", 0)
            table.append([label.format(n=n_ad), rr[tag].diag.get("routed_from"), adapted.format(n=n_ad),
                          fmt(rr[tag].acc, 2), fmt(ro[tag]["flip_vs_exact"], 2)])
    say(fmt_table(table, ["served as", "routed from", "adapted", "acc", "answers changed vs original"]))
    if "recentred" in rr:
        keys = ("readout", "blocks_executed", "routed_from", "reordered", "adapted", "adapt_n")
        say("diagnostics of one recentred decision: " + str({k: rr["recentred"].diag.get(k) for k in keys}))
    say(f"L0 under the rewording: {fmt(ro['l0_reworded'], 2)}\n")
    say("[3/4] Fewer labels", bold=True)
    perm = np.random.RandomState(args.seed).permutation(n_lab)
    curve = []
    for n in FAKE["budgets"]:
        d_n = Decider(be)
        art_n, t_n = timed(d_n.fit_head, q, [tr_s[i] for i in perm[:n]], [tr_y[i] for i in perm[:n]], layers=[block])
        decs, s = timed(d_n.decide_batch, te_s, q, level="L2")
        r = evaluate(decs, gold, q.options, s, nb)
        curve.append({"n_labels": int(n), "acc": r.acc, "ece": r.ece, "fit_seconds": t_n, "method": art_n["method"],
                      "oof_acc": art_n["cv"]["oof_acc"]})
    say(fmt_table([[c["n_labels"], fmt(c["acc"], 2), fmt(c["ece"], 2)] for c in curve], ["labels", "acc", "ece"])
        + "\n")
    say("[4/4] Artifact round trip", bold=True)
    rt = roundtrip(be, export, [(q, te_s, rows["L2"].probs)])
    say(roundtrip_line(rt, "export_artifacts"))
    auto = rt["decider"].decide(te_s[0], [q], level="auto")["route"].level
    say(f'level="auto": route -> {auto}; an unrelated question ("{UNRELATED.text}", 3 options) -> '
        f'{rt["auto_without_head"]}\n')
    seconds = time.perf_counter() - t_run
    say(f"done in {seconds:.1f} s.  Next: python -m demo.jev_mode   (a Qwen3 model + the shipped heads, real data)")
    fit = {"n_labels": n_lab, "fit_seconds": t_fit, "method": art["method"], "layer_abs": art["layer_abs"],
           "oof_acc": art["cv"]["oof_acc"], "acc": rows["L2"].acc, "acc_shipped": None, "acc_l0": rows["L0"].acc,
           "curve": curve}
    return {"backend": "fake", "model": be.name, "device": None, "dtype": None, "batch_size": None, "heads_file": None,
            "heads_block": block, "n_blocks": nb, "n_test": len(te_s), "reword": FAKE_REWORDING,
            "adapt_after": args.adapt_after, "fit_labels": n_lab, "seed": args.seed,
            "planted": {k: FAKE[k] for k in ("position_bias", "temperature", "logit_noise", "wording_shift",
                                             "layer_noise")},
            "levels": {"route": {"kind": q.kind, "K": q.k, "n": len(te_s),
                                 **{lv: r.to_dict() for lv, r in rows.items()}}},
            "routing": {"route": {**ro, "variant": "custom"}}, "fit": {"route": fit},
            "roundtrip": {k: rt[k] for k in ("n_heads", "bytes", "max_abs_dp", "n_decisions", "auto_without_head")},
            "seconds": seconds, "env": environment()}


# ---------------------------------------------------------------- the deployment lifecycle
def lifecycle_world(args):
    """(backend, question, stream states, stream labels, held-out states, held-out labels, reworded question, name,
    block or None). The fake world streams its 150 labelled tickets; the hf world streams the first --stream train
    cases of one typed question and holds out its --n-test test cases."""
    if args.backend == "fake":
        be, q, states, labels = fake_world(args.seed)
        n = min(FAKE["n_labelled"], args.stream)
        q_rew = Question.choice(FAKE_REWORDING, HANDLERS, name="route_reworded")
        held = FAKE["n_labelled"]
        block = args.block or FAKE["block"]
        return be, q, states[:n], labels[:n], states[held:], labels[held:], q_rew, "route", block
    name = resolve_question_names(args.questions)[0]
    be, t_load = timed(make_backend, args)
    say(f"model load {t_load:.1f} s")
    rec = typed_questions([name], args.n_test)[0]
    tr_s, tr_y = train_labels(args.stream)[rec.q.key]
    variant = args.reword if args.reword in REWORD_VARIANTS else "w2"
    return be, rec.q, tr_s[:args.stream], tr_y[:args.stream], rec.states, rec.labels, reworded(rec.q, name, variant), \
        name, args.block


def run_lifecycle(args) -> Dict[str, Any]:
    t_run = time.perf_counter()
    be, q, st_s, st_y, ho_s, ho_y, q_rew, name, block = lifecycle_world(args)
    gold = [q.options[y] for y in ho_y]
    nb = be.n_layers
    fit_kw = {"layers": [block]} if block else {}
    say(f"AnyJev Jev-mode demo, deployment lifecycle  |  backend {args.backend}"
        + (f" ({args.model})" if args.backend == "hf" else f" (synthetic model, {nb} blocks)")
        + f'  |  question "{q.text}" ({q.k} options)', bold=True)
    dec = Decider(be, level="auto", adapt_min_n=args.adapt_after)
    decs, s = timed(dec.decide_batch, ho_s, q)
    day0 = evaluate(decs, gold, q.options, s, nb)
    say(f"Day 0: no head. {len(ho_s)} held-out requests answered at {decs[0].level} (level auto): "
        f"acc {fmt(day0.acc, 2)}\n")
    say(f"Labelled requests arrive one at a time (Decider.observe, fit_at={args.fit_at}); the head solves itself:",
        bold=True)
    fits: List[Dict[str, Any]] = []
    rows = []
    for i, (st, y) in enumerate(zip(st_s, st_y), 1):
        art = dec.observe(q, st, y, fit_at=args.fit_at, **fit_kw)
        if art is None:
            continue
        decs, s = timed(dec.decide_batch, ho_s, q)
        r = evaluate(decs, gold, q.options, s, nb)
        fits.append({"at": i, "acc": r.acc, "ece": r.ece, "level": decs[0].level, "method": art["method"],
                     "layer_abs": art["layer_abs"], "oof_acc": art["cv"]["oof_acc"]})
        rows.append([i, "solved" if len(fits) == 1 else "re-solved", art["method"], f"{art['layer_abs']} of {nb}",
                     fmt(art["cv"]["oof_acc"], 2), f"{fmt(r.acc, 2)} ({decs[0].level})"])
    say(fmt_table(rows, ["labelled request", "head", "method", "block", "out-of-fold acc", "held-out acc (level)"]))
    say(f"held-out acc: {fmt(day0.acc, 2)} at L0 on day 0 -> {fmt(fits[-1]['acc'], 2) if fits else 'n/a'} after "
        f"{len(st_s)} labels, no call to fit_head by hand.\n")
    wording: Dict[str, Any] = {}
    if q_rew is not None:
        say(f'The question is reworded: "{q_rew.text}" (same options). Served by the same decider, one request at a '
            f"time:", bold=True)
        control = Decider(be, adapt=False)
        control.load_artifacts(dec.export_artifacts())
        first_n = min(args.adapt_after, len(ho_s))
        served, ctrl = [], []
        for st in ho_s:
            served.append(dec.decide(st, [q_rew])[0])
            ctrl.append(control.decide(st, [q_rew])[0])
        head_n = evaluate(served[:first_n], gold[:first_n], q.options, 0.0, nb).acc
        later = slice(first_n, None)
        adapted = evaluate(served[later], gold[later], q.options, 0.0, nb).acc if len(ho_s) > first_n else None
        ctrl_later = evaluate(ctrl[later], gold[later], q.options, 0.0, nb).acc if len(ho_s) > first_n else None
        d_last = served[-1].diagnostics
        say(f"routed to the head of \"{q.text}\" (routed_from={d_last.get('routed_from')}); first {first_n} requests "
            f"served as is: acc {fmt(head_n, 2)}; from request {first_n + 1} recentred on the unlabelled requests "
            f"so far: "
            f"acc {fmt(adapted, 2)} (the same requests without adaptation: {fmt(ctrl_later, 2)}); "
            f"adapt_n now {d_last.get('adapt_n')}.\n")
        wording = {"text": q_rew.text, "first_n": first_n, "acc_first_unadapted": head_n, "acc_after_adapted": adapted,
                   "acc_after_control": ctrl_later, "routed_from": d_last.get("routed_from"),
                   "adapt_n": d_last.get("adapt_n")}
    say("Restart:", bold=True)
    saved = dec.export_artifacts(include_observations=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "heads.json")
        json.dump(saved, open(path, "w"))
        size = os.path.getsize(path)
        dec2 = Decider(be, level="auto", adapt_min_n=args.adapt_after)
        dec2.load_artifacts(path)
    n_obs = len(saved.get("observations", {}).get(q.key, {}).get("labels", []))
    restart: Dict[str, Any] = {"n_heads": len(saved["heads"]), "n_observations": n_obs,
                               "n_adaptation": len(saved.get("adaptation", {})), "bytes": size}
    if q_rew is not None and len(ho_s) > first_n:
        again = [dec2.decide(st, [q_rew])[0] for st in ho_s[later]]
        dp = float(np.max(np.abs(np.stack([x.probs for x in again]) - np.stack([x.probs for x in served[later]]))))
        restart.update({"adapted_on_first_request": bool(again[0].diagnostics.get("adapted")),
                        "adapt_n_first": again[0].diagnostics.get("adapt_n"), "max_abs_dp": dp})
        say(f"export_artifacts(include_observations=True) -> {size / 1000:.0f} KB: {restart['n_heads']} head, {n_obs} "
            f"observations, adaptation statistics for {restart['n_adaptation']} question -> new Decider -> "
            f"load_artifacts -> the first reworded request is already adapted ({restart['adapted_on_first_request']}, "
            f"adapt_n {restart['adapt_n_first']}); max |dp| against the decisions before the restart: {dp:.2e}")
    else:
        say(f"export_artifacts(include_observations=True) -> {size / 1000:.0f} KB -> new Decider -> load_artifacts")
    seconds = time.perf_counter() - t_run
    say(f"\ndone in {seconds:.1f} s.")
    return {"backend": args.backend, "model": be.name, "mode": "lifecycle", "question": name, "seed": args.seed,
            "lifecycle": {"day0": {"level": day0.readout if day0.readout in ("raw", "L0", "L1") else "L0",
                                   "acc": day0.acc, "n": len(ho_s)},
                          "stream": len(st_s), "fit_at": args.fit_at, "fits": fits, "wording": wording,
                          "restart": restart},
            "seconds": seconds, "env": environment()}


# ---------------------------------------------------------------- the typed-decisions run
@dataclass
class Rec:
    name: str
    q: Question
    states: List[Any]
    labels: List[int]


def typed_questions(names: Sequence[str], n_test: int) -> List[Rec]:
    from bench.tasks import typed_decisions
    by_name = {f"{v[0].workflow}.{v[0].qname}": v
               for v in typed_decisions.group_by_question(typed_decisions.load("test")).values()}
    out = []
    for name in names:
        if name not in by_name:
            fail(f"question {name} is not in the dataset's test split")
        items = by_name[name][:n_test]
        out.append(Rec(name, items[0].question, [d.state for d in items], [d.gold_index for d in items]))
    return out


def train_labels(n: int) -> Dict[str, Tuple[list, list]]:
    """q.key -> (states, labels): the first n cases per workflow of the train split, one decision per case."""
    from bench.tasks import typed_decisions
    groups = typed_decisions.group_by_question(typed_decisions.load("train", limit_cases=n))
    return {k: ([d.state for d in v][:n], [d.gold_index for d in v][:n]) for k, v in groups.items()}


def estimate_line(sec_per_prompt: float, n_first: int, block: int, n_blocks: int, recs: Sequence[Rec], args) -> str:
    """Prompts still to run x measured seconds per prompt at each depth (the head's block and the full depth)."""
    at_block = full = 0
    for rec in recs:
        perms = 1 if rec.q.ordered else (2 if rec.q.kind == "noul" else rec.q.k)
        full += args.n_test * perms * (2 if args.reword != "none" else 1)
        at_block += 2 * args.n_test + (args.fit_labels + args.n_test if args.fit_labels else 0)
        if args.reword != "none":
            at_block += 2 * args.n_test + (args.n_test if rec.q.kind == "choice" else 0)
    secs = sec_per_prompt * (at_block + full * n_blocks / max(1, block))
    return (f"first batch ({n_first} states, block {block}) took {sec_per_prompt * n_first:.1f} s -> this run "
            f"({len(recs)} question{'s' if len(recs) != 1 else ''}, {args.n_test} states, {args.fit_labels} labels) "
            f"will take about {fmt_duration(secs)}; --n-test and --fit-labels shorten it, a GPU runs all 20 questions "
            "in ~3 min.")


def reference_lines(info: Optional[HeadsInfo], model: str) -> List[str]:
    """Reference numbers with their JSON sources, printed only when the files exist for this model."""
    out = []
    tp = (info.data.get("validation", {}).get("summary") or {}).get("typed_pooled") if info else None
    if tp:
        line = (f"Expected on this model: L2 pooled {tp['acc']:.3f} / ece {tp['ece']:.3f} "
                f"({short(info.path)}, validation.summary)")
        p = os.path.join(ROOT, "bench", "results_typed_v01", "2026-09-22", model.replace("/", "__") + ".json")
        l0 = (json.load(open(p)).get("levels", {}).get("L0", {}).get("overall", {}).get("acc")
              if os.path.exists(p) else None)
        out.append(line + (f"; L0 {l0:.3f} ({short(p)})" if l0 is not None else "")
                   + ". bf16 batch noise moves cells by ~0.01.")
    p = os.path.join(ROOT, "bench", "results_exit", "2026-09-22", "Qwen__Qwen3-8B.labels.json")
    if os.path.exists(p):
        rows = json.load(open(p)).get("rows", {})
        cells = [f"{n} -> {rows[f'head@{n}']['acc']:.3f}" for n in (20, 50, 100, 300) if f"head@{n}" in rows]
        out.append(f"Label budget measured on Qwen3-8B ({short(p)}): {', '.join(cells)}.")
    return out


def run_typed(args) -> Dict[str, Any]:
    t_run = time.perf_counter()
    info: Optional[HeadsInfo] = args.heads_info
    names = resolve_question_names(args.questions)
    be, t_load = timed(make_backend, args)
    nb = be.n_layers
    block = args.block or (info.block if info else None)
    dec_serve = Decider(be)
    if info:
        load_heads(dec_serve, info)
    say(f"AnyJev Jev-mode demo  |  {args.model} on {args.device} ({args.dtype}, batch {args.batch_size})  |  "
        + (f"heads {short(info.path)}: {info.n_heads} heads, block {info.block} of {info.n_blocks}, "
           f"built {info.built}" if info else "no heads file (fit-only)"), bold=True)
    say(f"LocalLLaMA/typed-decisions: {len(names)} question{'s' if len(names) != 1 else ''} x {args.n_test} held-out "
        f"test states (test split, first {args.n_test} per question). Train split used only for [3/4].")
    say(f"model load {t_load:.1f} s")
    recs = typed_questions(names, args.n_test)
    train = train_labels(args.fit_labels) if args.fit_labels else {}
    early_done: List[bool] = []
    estimate_done = args.device == "cuda"
    L0: Dict[str, Row] = {}
    L2: Dict[str, Row] = {}
    RO: Dict[str, Dict[str, Row]] = {}
    FIT: Dict[str, Tuple[Row, Dict[str, Any]]] = {}
    shipped_of: Dict[str, bool] = {}
    new_heads: Dict[str, Any] = {"model": be.name, "artifacts": {}, "heads": {}}
    rt_items: List[Tuple[Question, list, np.ndarray]] = []
    for rec in recs:
        q, states, gold = rec.q, rec.states, [rec.q.options[y] for y in rec.labels]
        shipped = shipped_of[rec.name] = info is not None and q.key in info.data.get("heads", {})
        if not shipped and not args.fit_labels:
            say(f"   {rec.name:48s} no shipped head and --fit-labels 0: skipped")
            continue
        dec_fit = art = None
        if args.fit_labels:                                        # [3/4] fit (first, so fit-only runs can serve it)
            if q.key not in train:
                fail(f"no train states for {rec.name}")
            s_tr, y_tr = train[q.key]
            dec_fit = Decider(be)
            art, t_fit = timed(dec_fit.fit_head, q, s_tr, y_tr, layers=[block] if block else None)
            warn_early_stop(dec_fit, early_done)
            if not estimate_done:
                estimate_done = True
                say(estimate_line(t_fit / len(s_tr), len(s_tr), art["layer_abs"], nb, recs, args))
            new_heads["heads"].update(dec_fit.export_artifacts()["heads"])
        serve = info.data if shipped else dec_fit.export_artifacts()
        dec_l2 = dec_serve if shipped else dec_fit
        decs, s = timed(Decider(be).decide_batch, states, q, level="L0")      # [1/4]
        L0[rec.name] = evaluate(decs, gold, q.options, s, nb)
        if not estimate_done:                                      # cpu/mps: the estimate after the first batch
            first, s1 = timed(dec_l2.decide_batch, states[:args.batch_size], q, level="L2")
            estimate_done = True
            say(estimate_line(s1 / len(first), len(first), first[0].diagnostics["blocks_executed"], nb, recs, args))
            rest, s2 = timed(dec_l2.decide_batch, states[args.batch_size:], q, level="L2")
            decs, s = list(first) + list(rest), s1 + s2
        else:
            decs, s = timed(dec_l2.decide_batch, states, q, level="L2")
        warn_early_stop(dec_l2, early_done)
        L2[rec.name] = evaluate(decs, gold, q.options, s, nb)
        RO[rec.name] = routing_rows(be, serve, q, reworded(q, rec.name, args.reword), states, gold, args.adapt_after,
                                    nb)
        if dec_fit is not None:                                    # [3/4] eval
            row = L2[rec.name]
            if shipped:
                decs, s = timed(dec_fit.decide_batch, states, q, level="L2")
                row = evaluate(decs, gold, q.options, s, nb)
            FIT[rec.name] = (row, {"n_labels": len(s_tr), "fit_seconds": t_fit, "method": art["method"],
                                   "layer_abs": art["layer_abs"], "oof_acc": art["cv"]["oof_acc"], "acc": row.acc,
                                   "acc_shipped": L2[rec.name].acc if shipped else None, "acc_l0": L0[rec.name].acc})
        rt_items.append((q, states, FIT[rec.name][0].probs if rec.name in FIT else L2[rec.name].probs))
        say(f"   {rec.name:48s} L0 {L0[rec.name].acc:.3f}  L2 {L2[rec.name].acc:.3f}"
            + (f"  reworded {RO[rec.name]['as_is'].acc:.3f} -> recentred {RO[rec.name]['recentred'].acc:.3f}"
               if "as_is" in RO[rec.name] else "")
            + (f"  head({args.fit_labels}) {FIT[rec.name][0].acc:.3f}" if rec.name in FIT else ""))
    if not L2:
        fail("nothing was run")
    by_name = {r.name: r for r in recs}
    # ---- [1/4]
    say(f"\n[1/4] {'Shipped' if any(shipped_of.values()) else 'New'} heads: L2 against the zero-label level on the "
        "held-out states", bold=True)
    p0, p2 = pooled(list(L0.values())), pooled(list(L2.values()))
    table = [[n, by_name[n].q.kind, by_name[n].q.k, fmt(a.acc), fmt(b.acc), fmt(a.ece, 2), fmt(b.ece, 2), a.prompts,
              f"{b.blocks} of {b.n_blocks}", f"{a.ms:.1f}", f"{b.ms:.1f}", b.readout]
             for n, (a, b) in ((n, (L0[n], L2[n])) for n in L2)]
    table.append([f"pooled ({p2['n']} decisions)", "", "", fmt(p0["acc"]), fmt(p2["acc"]), fmt(p0["ece"]),
                  fmt(p2["ece"]), "", "", "", "", ""])
    say(fmt_table(table, ["question", "kind", "K", "L0 acc", "L2 acc", "L0 ece", "L2 ece", "L0 prompts", "L2 blocks",
                          "L0 ms", "L2 ms", "L2 readout"]))
    say(f"ms = wall clock of decide_batch per state on {args.device}, tokenisation included (L0 pays K prompts through "
        f"all {nb} blocks, L2 one prompt stopped at the head's block). ece: 15 equal-mass bins, printed for n >= 50.")
    refs = reference_lines(info, args.model)
    if refs:
        say(refs[0])
    if info:
        texts = {r.q.text for r in recs}
        extra = [h.get("question_id") for h in info.data["heads"].values() if h.get("text") not in texts]
        if extra:
            say(f"{len(extra)} more heads in the file ({', '.join(map(str, extra))}): loaded, not exercised here "
                "(their datasets run through bench.run).")
    levels = {n: {"kind": by_name[n].q.kind, "K": by_name[n].q.k, "n": L2[n].n, "shipped_head": shipped_of[n],
                  "L0": L0[n].to_dict(), "L2": L2[n].to_dict()} for n in L2}
    levels["pooled"] = {"n": p2["n"], "L0": {"acc": p0["acc"], "ece": p0["ece"]},
                        "L2": {"acc": p2["acc"], "ece": p2["ece"]}}
    # ---- [2/4]
    routing: Dict[str, Any] = {}
    if args.reword != "none":
        variant = args.reword if args.reword in REWORD_VARIANTS else "custom"
        say(f"\n[2/4] The same question, reworded ({variant}) and re-listed, no new labels", bold=True)

        def cell(rr: Dict[str, Row], k: str, ref: Row) -> str:
            return f"{rr[k].acc:.3f} ({rr[k].flip_vs(ref):.2f})" if k in rr else "-"

        table, n_ad = [], 0
        for n, rr in RO.items():
            q_rew = reworded(by_name[n].q, n, args.reword)
            routing[n] = {**routing_json(rr, L2[n], q_rew), "variant": variant}
            table.append([n, fmt(L2[n].acc), cell(rr, "as_is", L2[n]), cell(rr, "recentred", L2[n]),
                          cell(rr, "reversed", L2[n]), fmt(rr["l0_reworded"].acc) if "l0_reworded" in rr else "-"])
            n_ad = max(n_ad, int(rr["recentred"].diag.get("adapt_n", 0))) if "recentred" in rr else n_ad
            if len(recs) == 1 and q_rew is not None:
                table.append(["  reworded text: " + q_rew.text.replace("\n", " / ")[:90], "", "", "", "", ""])
        pool = {k: pooled([rr.get(k) for rr in RO.values()]) for k in ("as_is", "recentred", "reversed", "l0_reworded")}
        table.append(["pooled", fmt(p2["acc"]), fmt(pool["as_is"]["acc"]), fmt(pool["recentred"]["acc"]),
                      fmt(pool["reversed"]["acc"]) + " (choice only)", fmt(pool["l0_reworded"]["acc"])])
        say(fmt_table(table, ["question", "exact head", "reworded, as is",
                              f"reworded, recentred on {args.n_test} unlabelled reqs", "options reversed, remapped",
                              "L0 under the rewording"]))
        say('(x) = share of states whose answer differs from the exact head\'s. "-" = no route (o1 changes the option '
            "texts) or a score/noul question, which keeps its order (ordinal bins; noul has no listing to reverse).")
        say(f'Rows are label-free: "recentred" is Decider(adapt="routed", adapt_min_n={args.adapt_after}) after '
            f"the {args.n_test} states were served once as unlabelled requests; the row is the second pass over the "
            f"same states (adapt_n {n_ad} on the last decision). --lifecycle shows the first pass, request by request.")
        say("Measured on Qwen3-4B/8B (bench/results_paraphrase/2026-09-22/*.b24.json): as is 0.65-0.70, recentred on "
            "30 states 0.74-0.75, labelled refit 0.77. Nothing here is a promise for this model; it prints what it "
            "measures.")
        routing["pooled"] = {"exact": p2["acc"], **{k: v["acc"] for k, v in pool.items()}}
    # ---- [3/4]
    fit: Dict[str, Any] = {n: f for n, (_, f) in FIT.items()}
    if FIT:
        say(f"\n[3/4] A new head from {args.fit_labels} labels (train split, block "
            f"{block if block else 'chosen by CV'})", bold=True)
        pf = pooled([r for r, _ in FIT.values()])
        ps = pooled([L2[n] for n in FIT if shipped_of[n]])
        table = [[n, f["n_labels"], f"{f['fit_seconds']:.1f}", f["method"], fmt(f["oof_acc"], 2), fmt(f["acc"]),
                  fmt(f["acc_shipped"]), fmt(f["acc_l0"])] for n, f in fit.items()]
        table.append(["pooled", "", "", "", "", fmt(pf["acc"]), fmt(ps["acc"]),
                      fmt(pooled([L0[n] for n in FIT])["acc"])])
        say(fmt_table(table, ["question", "labels", "fit s", "method", "oof acc", f"acc, {args.fit_labels}-label head",
                              "acc, shipped head (300)", "acc, L0"]))
        say(f"fit s = one forward of the {args.fit_labels} labelled states stopped at the block + the closed-form "
            "solve and 5-fold CV on CPU.")
        if len(refs) > 1:
            say(refs[-1])
        fit["pooled"] = {"acc": pf["acc"], "ece": pf["ece"], "acc_shipped": ps["acc"]}
    # ---- [4/4]
    say("\n[4/4] Artifact round trip", bold=True)
    export = new_heads if new_heads["heads"] else dec_serve.export_artifacts()
    rt = roundtrip(be, export, rt_items)
    say(roundtrip_line(rt, f"export_artifacts() of the {'new' if new_heads['heads'] else 'shipped'} heads"))
    if args.save_head:
        with open(args.save_head, "w") as f:
            json.dump(export, f)
        say(f"wrote {args.save_head}")
    say(("" if args.save_head else "--save-head not given; pass it to keep the file.  ")
        + f'level="auto" on a question with no head -> {rt["auto_without_head"]} (checked).')
    seconds = time.perf_counter() - t_run
    say(f"done in {fmt_duration(seconds)}." + (f"  JSON: {args.json}" if args.json else ""))
    return {"backend": "hf", "model": args.model, "device": args.device, "dtype": args.dtype,
            "batch_size": args.batch_size, "heads_file": short(info.path) if info else None,
            "heads_block": block, "n_blocks": nb, "n_test": args.n_test, "reword": args.reword,
            "adapt_after": args.adapt_after, "fit_labels": args.fit_labels, "model_load_seconds": t_load,
            "levels": levels, "routing": routing, "fit": fit,
            "roundtrip": {k: rt[k] for k in ("n_heads", "bytes", "max_abs_dp", "n_decisions", "auto_without_head")},
            "seconds": seconds, "env": environment(batch_size=args.batch_size)}


# ---------------------------------------------------------------- single state
def run_single_state(args) -> Dict[str, Any]:
    t_run = time.perf_counter()
    if args.state_file.endswith(".json"):
        obj = json.load(open(args.state_file))
        text, kind = render_state(obj), type(obj).__name__
    else:
        obj = text = open(args.state_file).read()
        kind = "text"
    info = heads_info(args)
    be = make_backend(args)
    nb = be.n_layers
    dec = Decider(be, adapt_min_n=args.adapt_after)
    if info:
        load_heads(dec, info)
    if args.question in TYPED_QUESTIONS:
        if info is None:
            fail(f"a shipped question name needs a heads file for {args.model} (shipped: {SHIPPED_MODELS})")
        q, source = shipped_question(info, args.question), "shipped"
    else:
        q, source = parse_question_spec(args.question), "own"
    block = args.block or (info.block if info else None)
    fit_rec = None
    if args.labels_file:
        rows = json.load(open(args.labels_file))
        need = max(8, 2 * q.k)
        if len(rows) < need:
            fail(f"--labels-file {len(rows)} is below max(8, 2K) = {need} for {q.id} (K={q.k})")
        art, t_fit = timed(dec.fit_head, q, [r["state"] for r in rows], [int(r["label"]) for r in rows],
                           layers=[block] if block else None)
        fit_rec = {"n_labels": len(rows), "fit_seconds": t_fit, "method": art["method"], "layer_abs": art["layer_abs"],
                   "oof_acc": art["cv"]["oof_acc"]}
        say(fit_line(art, t_fit))
        if args.save_head:
            dec.save_artifacts(args.save_head)
            say(f"wrote {args.save_head}")
    pre, suf = render_chat_parts(be.tokenizer, build_prompt(text, q, list(range(q.k))))
    n_tok = len(be.tokenizer.encode(pre + suf, add_special_tokens=False))
    routed = dec.route(q)
    kind_of_head = "shipped head" if source == "shipped" else ("head fit now" if fit_rec else "routed head")
    head_desc = "no head" if routed is None else f"{kind_of_head}, block {routed[0]['layer_abs']} of {nb}"
    say(f"state: {os.path.basename(args.state_file)} ({kind}, {len(text):,} chars, {n_tok} prompt tokens)   "
        f"question: {q.id} ({q.kind}, {q.k} options; {head_desc})")
    if args.backend == "fake":
        say("synthetic model: your state has no planted content, the distribution is meaningless")
    say()
    d, t_main = timed(lambda: dec.decide(obj, [q], level="auto")[0])
    d0, t_l0 = (timed(lambda: Decider(be).decide(obj, [q], level="L0")[0]) if d.level == "L2" else (d, t_main))
    dg, perms0 = d.diagnostics, int(d0.diagnostics.get("permutations", 1))
    n_pr = int(dg.get("permutations", 1))
    cols = [f"{d.level} ({n_pr} prompt{'s' if n_pr != 1 else ''}, {dg.get('blocks_executed', nb)} blocks, "
            f"{1000 * t_main:.0f} ms)"]
    if d.level == "L2":
        cols.append(f"L0 ({perms0} prompts, {nb} blocks, {1000 * t_l0:.0f} ms)")
    say(fmt_table([[o[:80], f"{bar(float(d.probs[i]))} {d.probs[i]:.2f}"]
                   + ([f"{d0.probs[i]:.2f}"] if d.level == "L2" else []) for i, o in enumerate(q.options)],
                  ["option"] + cols))
    detail = (f"{dg.get('readout')}, temperature {dg.get('temperature', 0):.2f}" if d.level == "L2"
              else f"prior {dg.get('prior_method')}")
    flips = int(round(float(d0.diagnostics.get("order_flip_l0", 0.0)) * max(0, perms0 - 1)))
    l0_note = (f"   L0 {'agrees' if d0.argmax == d.argmax else 'says ' + d0.argmax} (its answer changes under "
               f"{flips}/{perms0} cyclic shifts)" if d.level == "L2" else "")
    say(f"answer: {d.answer}  (level {d.level}, confidence {d.confidence:.2f}, {detail}){l0_note}")
    if d.level != "L2":
        say("routing: no head, answered at L0 -- pass --labels-file to fit one")
    elif dg.get("routed_from") and dg.get("routed_from") != q.id:
        seen = int(dg.get("adapt_n", 0))
        say(f"routing: routed from {dg['routed_from']} under another wording; " + (
            f"recentred on {seen} requests" if dg.get("adapted") else
            f"recentring starts after {args.adapt_after} requests (seen {seen})"))
    else:
        say("routing: exact head.")
    seconds = time.perf_counter() - t_run
    say(f"done in {seconds:.1f} s.")
    return {"backend": args.backend, "model": be.name, "device": args.device if args.backend == "hf" else None,
            "state_file": args.state_file, "state_chars": len(text), "prompt_tokens": n_tok,
            "question": {"id": q.id, "kind": q.kind, "text": q.text, "options": list(q.options), "source": source},
            "decision": d.to_dict(), "diagnostics": pick(d), "ms": 1000 * t_main,
            "l0": {"decision": d0.to_dict(), "diagnostics": pick(d0), "ms": 1000 * t_l0}, "fit": fit_rec,
            "seconds": seconds, "env": environment()}


# ---------------------------------------------------------------- entry point
def main(argv=None) -> int:
    args = parse_args(argv)
    mode = preflight(args)
    result = {"single": run_single_state, "fake": run_fake, "typed": run_typed, "lifecycle": run_lifecycle}[mode](args)
    write_json(args.json, result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:                       # usage / prerequisite errors carry a message: exit code 2
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise
