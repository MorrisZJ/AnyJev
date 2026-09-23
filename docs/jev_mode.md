# Jev mode: a closed-form head at two thirds of the depth

Jev is a hosted model built to answer typed questions (pick one, yes or no, a score) cheaply and
with a probability. AnyJev's first two levels read those answers off any open model with no
labels (L0) or a temperature (L1). This page is the third level, **L2**, and the shipped
artifacts that come with it: for every supported model, a set of per-question heads for the
LocalLLaMA/typed-decisions questions, plus the recipe to add your own question in
seconds with 100-300 labels.

The whole method is one sentence: keep the model, stop its forward at about two thirds of the
depth, and read the decision from that hidden state with a `[hidden, K]` matrix solved in closed
form on labelled examples. No gradient, no new model weights, no second model.

## Numbers

LocalLLaMA/typed-decisions, 20 questions, 300 labelled decisions per question to fit, 100 held
out per question to test (2000 test decisions pooled). The block is chosen on the calibration
data alone (`bench.jev_mode_table`, entry 8 of `docs/research_log.md`); the cost column is the
measured ms per decision on one H100 NVL for 110- and 1000-token states, batched
(`bench.exit_latency`, `bench/results_exit/2026-09-22/<model>.latency.json`).

| model | raw | L0 | L1 | **L2, fixed block** | block | L2 at full depth | cost of L2 vs one plain forward |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.468 | 0.494 | 0.499 | **0.730** | 18 of 28 (64%) | 0.738 | 0.70x time (1000-token states; launch-bound at 110), 0.64x FLOPs |
| Qwen3-4B | 0.547 | 0.564 | 0.567 | **0.786** | 24 of 36 (67%) | 0.773 | 0.67x / 0.69x time (110 / 1000-token states), 0.67x FLOPs |
| Qwen3-8B | 0.626 | 0.647 | 0.648 | **0.771** | 24 of 36 (67%) | 0.767 | 0.68x / 0.68x time, 0.67x FLOPs |
| Qwen3-30B-A3B (MoE) | 0.599 | 0.630 | 0.630 | **0.799** | 40 of 48 (83%) | 0.789 | not measured (eager MoE) |
| Qwen3-32B | 0.684 | 0.700 | 0.699 | **0.798** | 52 of 64 (81%) | 0.795 | 0.82x / 0.84x time, 0.81x FLOPs |
| Jev (published) | | | | 0.727 | | | hosted, per request |
| Laya, fine-tuned 421M (measured) | | | | 0.768 | | | |

Pooled ECE of L2 is 0.03-0.05 on every model (L1: 0.036-0.055; on the 32B L2 is 0.046 against
L1's 0.036; L0: 0.15-0.40). Bootstrap 95% CIs are about +-0.02 wide; every fixed-block cell is
within its CI of the full-depth head on all five models. Flip under option reversal of the L2
head is 0.06-0.11 (`<model>.jevmode.json`). The cost column is the batched ms per decision on one
H100 (a quiet GPU host, load average 26) at the fixed block relative to the model's plain forward; the truncated loop at
full depth costs the same as the plain forward, so the ratio is the depth fraction plus launch
overhead.

How to read it. A 1.7B model at 64% of its depth reaches 0.730 on the test split (95% CI
[0.710, 0.750]); the 0.727 listed for Jev (Laya's BENCHMARKS.md, `bench/run_typed.py`) was not
measured here, so the two are comparable as a level, not inside a CI. A 4B at 67% lands at 0.786
against the fine-tuned Laya's 0.768 - inside the 95% CI [0.767, 0.804], so a tie - with the same
300 labels per question and about ten times Laya's 421M parameters; it is above every zero-label
number we have from any size, including the 32B's L0 (0.700). The 32B and the 30B-A3B at 81-83%
give the highest numbers here (0.798-0.799): the same weights, one prompt per state, a fifth
of the forward skipped, +0.10 and +0.17 over their own zero-label readouts. The gain is largest
where the model is weakest (L1 to L2: +0.23 on the 1.7B, +0.22 on the 4B, +0.12 on the 8B, +0.17
on the 30B-A3B, +0.10 on the 32B), and L2 costs
less than L0 or L1, not more: one prompt per state instead of K cyclic shifts, and the forward
stops early. The 32B does not truncate as far as the smaller models (its decisions are not
readable before block 48 of 64), so when cost matters the 4B head, at about a sixth of the 32B's
wall clock (29.9 vs 194.6 ms batched at 1000-token states, 5.2 vs 34.1 at 110; a tenth of its
FLOPs), is the better trade, and when the last point of accuracy matters the 32B head is.

Label budget (Qwen3-8B, entry 5 of the research log, `Qwen__Qwen3-8B.labels.json`, block chosen
among 22 / 28 / 36): 20 labels give 0.654 (95% CI [0.641, 0.667]), level with L0 / L1 in the full
typed run (0.647 / 0.648) - labels start paying at about 50, which gives 0.707; 100 give 0.740
(against the 0.727 listed for Jev), 200 give 0.754, 300 give 0.772. Calibration needs more
labels than accuracy: pooled ECE is 0.11 at 50 labels and 0.03 at 300.

## Use it

Run the demo first (`demo/jev_mode.py`; the CLI is documented in the README's "Try it in 60 seconds"):

```
python -m demo.jev_mode --backend fake      # seconds, no download: fit, route, recentre, reverse, export, reload
python -m demo.jev_mode --backend fake --lifecycle   # the deployment lifecycle: day 0 at L0, observe() solves the head at
                                                     # 30 labels, a rewording recentred from traffic, export / restart / load
python -m demo.jev_mode                     # Qwen3-1.7B + the shipped heads on typed-decisions (GPU, or a slow CPU)
python -m demo.jev_mode --state-file ticket.json --question customer_service.category   # one state, a shipped head
```

Then the API:

```python
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

dec = Decider(HFBackend("Qwen/Qwen3-4B"))
dec.load_artifacts("anyjev-heads/Qwen__Qwen3-4B.json")     # the shipped heads for the typed questions

q = Question.choice("What should happen next?", ["resolve", "escalate", "wait", "refund"], name="action")
d = dec.decide(ticket_text, [q], level="L2", require="L2")["action"]
d.probs, d.argmax, d.diagnostics["blocks_executed"]        # calibrated probabilities, forward stopped early
```

Your own question, 100-300 labelled states:

```python
art = dec.fit_head(q, states, labels)          # one forward of the states + a closed-form solve; seconds
json.dump(art, open("my_question.json", "w"))  # per (model, question); dec.load_artifact(q, art) restores it
```

Or let the head grow out of the traffic. Every request whose correct answer comes back later (a review
queue, an outcome, the LLM being replaced) is one call, and the decider fits and refits by itself:

```python
dec = Decider(HFBackend("Qwen/Qwen3-4B"), level="auto")
art = dec.observe(q, ticket_text, label)       # None until 30 observations exist; then the artifact dict of the head it solved
```

`observe(question, state, label, fit_at=30, refit_factor=2.0, **fit_kwargs)` records the pair and calls
`fit_head` on every pair recorded so far once `fit_at` of them exist (never below max(8, 2K)), then again
each time the count grows by `refit_factor` (30, 60, 120, ...); it returns the artifact dict on the call
that (re)solved the head, else `None`. With `level="auto"` the question answers at L0 until the first
solve and at L2 afterwards; nobody decides when to fit. `observations(q)` returns the (states, labels)
recorded so far, and `export_artifacts(include_observations=True)` writes them next to the heads.
`python -m demo.jev_mode --backend fake --lifecycle` plays this through in a second on the synthetic model
(its numbers are planted, not measured on a model): 50 held-out requests at L0 on day 0, acc 0.86; the head
solves itself at the 30th label (held-out 0.98 at L2, out-of-fold 0.97), re-solves at 60 and 120 (1.00); a
rewording is routed and served as is for 30 requests (0.47), then recentred (1.00; 0.85 for the same
requests without adaptation); export, restart, load: the first reworded request is already adapted and the
probabilities are identical to floating-point precision (max |dp| under 1e-50). `--backend hf --questions <workflow.qname>` runs the same script on a real
model with that question's train split as the stream (`--stream`, `--fit-at`).

`fit_head` shows each calibration state its options in a random order when there are at most 8
(`listing="auto"`, so the head reads the answer wherever it is listed), tries shrunk LDA and ridge on the candidate
blocks (default 50 / 60 / 70 / 85 / 100 % of depth) and keeps the one with the best out-of-fold
NLL; its temperature is fit on the out-of-fold scores. `calibrate(q, states, labels, level="L2")` is the same call. The artifact is
`Decider.export_artifacts()`'s `heads` entry: `W` (hidden x K, float32), `b`, the standardisation
`mean` / `scale`, `temperature`, the block index, the head kind, `n_calib` and the CV numbers. The same
export carries, per routed question, the running `sum` / `sumsq` / `n` of its features (`adaptation`, arrays in
the compact format) and, with `include_observations=True`, the labelled states recorded by `observe`;
`load_artifacts` restores all of it. On Qwen3-8B one head is 4096 x K + 2 x 4096 float32 values; a model's set of 20 typed heads plus
3 bench heads is 1.8-4.4 MB in the shipped format (arrays as base64 float32, exact; plain lists
are read too).

What a decision at L2 carries: `level == "L2"`, `diagnostics["readout"]` (`head:lda` or
`head:ridge`), `exit_layer` / `blocks_executed` / `n_blocks` / `relative_depth`, `temperature`,
`n_calib`. `require="L2"` raises `LevelError` on anything lower.

## Shipped artifacts

`anyjev-heads/<model>.json`, built by `scripts/build_heads.py` with `Decider.fit_head` on the real
backend and validated through the same `decide_batch(level="L2")` path a user runs (the build
also applies each head to the study cache's features and records the agreement, so the shipped
path is known to read the same states the studies did). Each file carries the per-question and
pooled validation numbers, the candidate blocks, `anyjev.__version__` and the `env` block (torch,
transformers, GPU, batch size, git commit).

| model | heads | block | typed pooled acc (live) | typed pooled ECE | banking20 / newsgroups / injection | ms per decision (live, batch 16) | file |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 20 typed + 3 bench | 18 of 28 | 0.728 | 0.035 | 0.785 / 0.585 / 0.960 | 5.9 | `anyjev-heads/Qwen__Qwen3-1.7B.json` (1.8 MB) |
| Qwen3-4B | same | 24 of 36 | 0.784 | 0.037 | 0.850 / 0.630 / 0.970 | 10.5 | `anyjev-heads/Qwen__Qwen3-4B.json` (2.2 MB) |
| Qwen3-8B | same | 24 of 36 | 0.763 | 0.037 | 0.875 / 0.685 / 0.995 | 14.5 | `anyjev-heads/Qwen__Qwen3-8B.json` (3.5 MB) |
| Qwen3-30B-A3B (MoE) | same | 40 of 48 | 0.794 | 0.039 | 0.865 / 0.755 / 0.995 | 234 (eager MoE, batch 8) | `anyjev-heads/Qwen__Qwen3-30B-A3B-Instruct-2507.json` (1.8 MB) |
| Qwen3-32B | same | 52 of 64 | 0.795 | 0.051 | 0.840 / 0.735 / 0.995 | 63.3 (batch 8) | `anyjev-heads/Qwen__Qwen3-32B.json` (4.4 MB) |

The typed heads (K <= 5) are fit on random listing orders, the K = 20 bench heads on the canonical
order (`listing="auto"`). Every head in a file reads the same block. Each (state, question) pair is still its own prompt and
its own truncated forward (the prompt carries the question and its options), but all of them stop
at the same depth, so they batch into one call and a state's prefix KV can be shared across the
questions asked of it. The live numbers are within bf16 batch noise of the cache-replayed table
above (the 4B reads 0.784 live against 0.786 replayed); the agreement check against the study
caches is |dp| = 0.0000 for the 1.7B / 4B / 30B-A3B / 32B (same extraction code) and 0.008 for the 8B
(older cache; `cache_mean_abs_dp` in the file). The ms column is the wall clock of `decide_batch` over the 100 test states of a
question, including tokenisation, for 300-400-token typed states. Validation JSON per model:
`bench/results_exit/2026-09-22/<model>.artifact.json`; the per-question-CV builds (each question
picking its own block among four candidates) are 0.741 / 0.767 / 0.775 at 4-5 blocks deeper, so
the fixed block is the better product (`<model>.artifact_cv.json`).

## The same question, reworded or re-listed: routing and label-free adaptation

A head is fit for one layout, but the question it serves rarely arrives in exactly that form.
Two things were measured (entry 13 of the research log; `bench.paraphrase_study`, 20 typed
questions, three hand-written rewordings of each question and one of its option descriptions,
Qwen3-4B and Qwen3-8B, 2000 test decisions per row):

| Qwen3-8B, block 24 (`bench/results_paraphrase/2026-09-22/Qwen__Qwen3-8B.paraphrase.b24.json`, `.order.json`) | head as is | + recentred on 30 unlabelled states | + recentred on 300 | refit with 300 labels |
|---|---|---|---|---|
| original wording | 0.767 | | | |
| question reworded (three ways) | 0.646-0.701 | 0.742-0.749 | 0.744-0.758 | 0.770 |
| option descriptions reworded | 0.745 | 0.754 | 0.757 | 0.765 |
| options listed in reverse, head fit on one order | 0.672 | | 0.729 (recentred on the 100 test states) | |
| options listed in reverse, head fit on random orders | 0.760 | | | |

(The 30-state cells are the mean of three draws over the 100 test states; the other cells are one pass over
the 2000 test decisions.)

- **A rewording moves the hidden states mostly by a shift and a rescaling.** The head as is
  loses 7-12 points and changes its answer on 18-27% of the states on the 8B (26-32% on the 4B;
  the model's own zero-label answer changes on 10-20%). Re-estimating the head's feature mean and scale on the states the
  reworded question is asked on, with no labels, recovers to within 2-3 points of a labelled
  refit with 30 states (0.74-0.75) and within 1-2 with 300 (0.74-0.76); 10 states already give
  0.73. `Decider(adapt="routed")` (the
  default) does this automatically for any question served by another layout's head, once
  `adapt_min_n=30` states have been seen (the current batch counts); `adapt=True` applies it
  always (costs about a point on the head's own layout), `adapt=False` never.
- **A head fit on one listing order is bound to it**; fit on a random order per calibration
  state (`fit_head(listing="random")`, same labels, same one forward) it loses 1-3 points under
  reversal instead of 9-10 and flips 7% of answers instead of 18%
  (`<model>.order.json`). This holds up to
  a handful of options; at K = 20 (banking20, newsgroups) random listings cost the head 6-18
  points, so the default `listing="auto"` is random up to 8 options and canonical above.
- **Routing** (`Decider.route(q)`): the exact layout first; then the same kind and option texts
  under another wording; then, for random-listing heads, the same option set in another order
  (probabilities are mapped back by option text). Diagnostics say what happened: `routed_from`,
  `reordered`, `adapted`, `adapt_n`. `level="auto"` picks L2 where a head routes, else L1 where
  an artifact exists, else L0, so nothing ever fails for lack of a head.
- **The statistics survive a restart.** `export_artifacts()` writes, next to the heads, the running sum,
  sum of squares and count of every routed question's features (`adaptation`), and `load_artifacts`
  reads them back, so a restarted service answers a reworded question adapted from its first request,
  with the same probabilities as before (the lifecycle demo checks this on the synthetic model: max |dp|
  under 1e-50 across export / restart / load, i.e. identical to floating-point precision). `include_observations=True` adds the labelled states recorded
  by `observe`, so a later refit, or a new base model, can reuse them.

This is the sense in which the head adapts at test time - feature re-standardisation, no
gradients, `W` untouched: a head is solved from labels in seconds, then follows its question
across wordings and listings from unlabelled traffic alone; with
`observe` the labels themselves come from the loop, and the head solves and re-solves itself as they
arrive (30, 60, 120, ...). What does not follow: a different option set (a new K, a renamed option) or a genuinely
different question, which get their own head or L0.

## Distilling a big model's Jev mode into a small one

Closed-form distillation (`bench.distill_heads`, entry 14): the 32B's shipped heads (0.795 on
typed) label 1200 synthetic cases per workflow written by Qwen3-8B from three real exemplars
each (`bench.synth_states`), and a small model's head is solved on its 300 gold labels plus the
teacher-labelled states. Qwen3-1.7B: 0.730 -> 0.760 (CI [+0.015, +0.049]); Qwen3-4B and 8B,
already within 1-2 points of the teacher on their own labels, do not move. Soft targets from
the teacher on the same 300 states add at most a point. Synthetic states alone are 6-9 points
below the 300 real ones, so the teacher supplements labels rather than replacing them. No
gradient anywhere in the pipeline.

## What it is not

- **Not transferable across questions.** A head is a function of one question on one model
  (the wording and listing of that question are handled above). Heads fit on other questions do
  not help a new one: question-agnostic heads of every form we tried (dot, probe, conditional
  logit, residual, position, option-line product) score at or below L0 on held-out questions
  (entries 1 and 3 of the research log). That is exactly the criticism the README makes of
  Laya's trained head, and it applies here too; the difference is that an L2 head costs seconds
  and 100-300 labels, not a training run, and lives next to the model you already serve.
- **Not a zero-label level.** Labels must come from the task. The model's own answers (heads fit
  on self-labels score 0.625 against raw 0.626, `bench/results_heads/2026-09-22/Qwen__Qwen3-8B.typed.json`)
  and a 32B teacher's zero-label answers (entry 7: 300 teacher labels 0.688 < 100 gold labels 0.726,
  `bench/results_distill/2026-09-22/Qwen__Qwen3-8B.from.Qwen__Qwen3-32B.json`) both cap the student
  at the teacher's accuracy.
- **Not for API or vLLM backends.** L2 reads hidden states, so it needs the local transformers
  backend; the other backends stop at L1.
- **Not a cascade.** Confidence-gated early exit over heads at several depths was tried on the
  8B and the 32B and did not beat one fixed block: on the 8B the gate lands on the fixed block 22
  (0.769 at a mean of 21.5 blocks against 0.770 at block 22), on the 32B the best pooled rule is
  0.792 at 48.9 blocks against 0.798 at the fixed block 52 (entries 4 and 10 of the research log,
  `bench/results_exit/2026-09-22/<model>.cascade.json`). The code was removed in version 3; the JSON
  is kept.
- **Not reasoning.** One prefill, so the ceiling is what the model knows without thinking.
- **Not measured against human labels on typed-decisions.** The gold of that set is one teacher
  model's soft label per decision (`bench/tasks/typed_decisions.py`), so a score on it measures
  agreement with that teacher, not correctness: over the 2,000 decisions the gold option carries a
  mean teacher probability of 0.66 and only 2 of the 2,000 gold distributions put all their mass
  on one option (`gold_probs` in `bench/results_typed_dump/2026-09-21/<model>.items.json.gz`).
  Every head on this page is fit on 300 of those labels per question, so read the numbers as
  "reproduces that teacher". The three bench tasks carry their own datasets' labels.

## Reproduce

```
python -m bench.extract_pools --model Qwen/Qwen3-4B --layers 12,14,16,18,20,22,24,26,28,30,32,34,36 --out bench/results_exit   # block-loop caches (GPU, once)
python -m bench.exit_study --model Qwen/Qwen3-4B --n-layers 36   # accuracy vs depth
python -m bench.jev_mode_table --model Qwen/Qwen3-4B --n-layers 36   # the product table (CPU replay)
python -m bench.exit_latency --model Qwen/Qwen3-4B               # ms per decision by depth (GPU)
python scripts/build_heads.py --model Qwen/Qwen3-4B --layers 22,24,28,36   # the shipped artifact (GPU)
```

`scripts/exit_parity.py` checks the truncated block loop against the plain forward (all
deviations 0 in fp32 on Qwen3-1.7B).
