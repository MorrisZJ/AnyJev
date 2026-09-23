# Multimodal state

A state can carry pictures next to its text. Nothing above the prompt builder
changes: the same three typed questions, the same single-prefill readout, the
same cyclic-shift marginalization, the same L1 artifacts, the same mandatory
`level` on every `Decision`.

```python
from anyjev import Decider, Image, Question
from anyjev.backends.hf_vlm import VLMBackend

d = Decider(VLMBackend("Qwen/Qwen3-VL-2B-Instruct"))

q = Question.choice("What is the user's screen showing?",
                    ["a login form", "a payment page", "an error", "something else"],
                    name="screen")
r = d.decide({"screenshot": Image("shot.png"), "note": "user says it is stuck"}, [q])

r["screen"].distribution
r.level                      # "L0"
```

| to | see |
|---|---|
| install | `pip install "anyjev[vlm]"` (transformers ≥ 4.57 for Qwen3-VL) |
| put a picture in a state | [Putting an image in a state](#putting-an-image-in-a-state) |
| pick a prior | [Which prior to use](#which-prior-to-use-measured-and-it-depends-on-the-task) |
| fit L1 or an L2 head on pictures | [L2 on images](#l2-on-images) |
| run the benchmarks and the tests | [Running the benchmarks and the tests](#running-the-benchmarks-and-the-tests) |
| read the results, and where they are stored | [results_multimodal.md](results_multimodal.md#where-the-results-are) |

## Putting an image in a state

`anyjev.Image` takes a path, a URL, a `data:` URI, raw bytes, or a PIL image.
Nothing is decoded until a backend asks for it.

Images may sit anywhere in the state — on their own, as a dict value, or in a
chat transcript. `split_state` walks the structure, replaces each picture with
an `<image i>` marker in traversal order, and returns the text and the ordered
images separately. The readout then splits the user turn at those markers, so
a picture lands in the prompt where it sat in the state rather than all of
them bundled up front.

```python
Image("shot.png")                                   # on its own
{"screenshot": Image("shot.png"), "note": "..."}    # a field of a dict
[{"role": "user", "content": [                      # a chat turn
    {"type": "image", "image": "shot.png"},
    {"type": "text", "text": "what is wrong here?"}]}]
```

PIL images are recognized directly. A bare string is *not*: a path and a
sentence look the same to a walker, so strings must be wrapped in `Image` or
declared with a `{"type": "image", ...}` content part.

If your own text already contains something that looks like a marker — an
MMMU question that says "look at `<image 1>`", a note that mentions
`<image 2>` — nothing breaks: each picture is placed once, at the first marker
that names it, a repeat or a marker with no such picture stays as text, and
the backend receives the pictures in exactly the order their placeholders
appear (`anyjev.readout.prompt_images`).

A state with no pictures renders exactly as it did before there was a
`split_state`, and text-only backends are untouched.

## What the corrections mean when the state is pixels

**Position bias** is unchanged: the option list is still text, still shown in
K cyclic rotations, still combined in log space. Whether a vision model's
position bias is as large as a text model's is an empirical question, and the
bench reports it the same way — `flip` is still the fraction of items whose
answer changes when the option list is reversed.

**Prior correction** needed one decision. The content-free probe exists to ask
"what does this model answer when the input carries no information?", and with
a picture in the state, blanking only the text does not do that — the image
still carries the content. So the probe blanks *every* modality: the text
becomes `N/A` / empty / `[MASK]` as before, and each picture is replaced by a
flat grey image of a fixed size (`anyjev.media.CF_IMAGE_SIZE`). The probe
keeps the shape of a real prompt — same question, same number of images — so
the prior it measures is comparable to the real distribution it divides out.

The probe is cached per `(question, number of images)` rather than per
question, and shared across every state with that many pictures, so a
300-item bench pays for it once.

The batch prior needs no change: it is the mean prediction over real inputs
and does not care what those inputs are made of.

## Which prior to use: measured, and it depends on the task

The default does not change with the modality: L0 uses the batch prior, as on
text, and the multimodal bench reports that default exactly as the text bench
does, with the content-free prior as an ablation row from the same forward
passes ([results](results_multimodal.md)). What follows is guidance for
opting in. Position debiasing helped on every task; the content-free prior
did not behave like something you can switch on blind:

- **POPE** (is this object in the image?): the content-free prior is the
  largest win in the whole multimodal bench — 8B accuracy 0.840 → 0.900, ECE
  0.147 → 0.065. The batch prior does almost nothing, because the true Yes
  rate is exactly 0.5 and the model's *mean* prediction sits close to it: the
  Yes-lean only shows when the picture is blanked.
- **pets20** (which breed?): the batch prior helps and the content-free prior
  changes nothing.
- **AI2D** (per-item science-diagram questions): the content-free prior costs
  8 to 24 accuracy points. The question and its options often give the answer
  away without the diagram, so the blank-image probe measures what the model
  knows, and dividing that out throws the knowledge away.

The working rule for opting in: the content-free prior is only safe when the
text half of the prompt carries no answer on its own, and only useful when
there is a label prior to remove. It is a per-task choice — the bench prints
the row for free, so measure it rather than assume it.

**L1 has a limit here too.** Temperature scaling minimizes NLL. On POPE with
the content-free prior as L0, the remaining 4B and 8B errors are confident
hallucinations ("yes, there is a snowboard", at
0.999), which dominate the NLL, so the fitted temperature flattens every
answer to shrink them: test NLL falls (8B 0.637 → 0.371) while ECE *rises*
(0.065 → 0.147). One scalar cannot fix confident wrong answers, which is the
README's "calibration cannot fix a model that cannot answer" in a new form.

## Backends

The backend contract grows one optional argument:

```python
next_token_logprobs(prompts, token_ids, images=None)
```

`images[i]` is the ordered list of `Image` for prompt `i`, one per placeholder
the chat template rendered. A backend that handles them sets
`accepts_images = True`; the decider only passes the argument when a state
actually carries a picture, and raises a clear error if a state has images and
the backend does not. Text-only backends are unchanged.

Vision models keep their chat template on the processor rather than the
tokenizer, because only the processor knows how many tokens a picture expands
to. A backend can point at it with `chat_renderer`; the decider falls back to
the tokenizer.

The shared-prefix fast path (`score_shared`, one prefix forward per state and
K short suffixes) carries text only, so prompts with pictures always go
through `next_token_logprobs`; text prompts in the same call still share.
Content-free probes get their own forward call on either path.

`anyjev.backends.hf_vlm.VLMBackend` is the transformers implementation
(`AutoModelForImageTextToText`, so Qwen2.5-VL / Qwen3-VL, Gemma 3, LLaVA and
the rest load through the same path). It needs transformers ≥ 4.57 for
Qwen3-VL.

### One thing that would have been a silent bug

The text backend passes explicit `position_ids` so that left padding does not
shift positions. A vision model with multimodal rope derives its position ids
from the image grid, and handing it the text backend's positions corrupts
them. `VLMBackend` therefore does not pass `position_ids` and lets the model
compute them.

`scripts/smoke_vlm.py` checks this by running the same prompts at batch size 1
and at batch size 8, with image sizes and caption lengths chosen to differ so
that padding actually varies within a batch. On Qwen3-VL-2B-Instruct the
answer distributions agree to `5.6e-10` and the argmax agrees 9/9. The raw
log-probabilities differ by up to 0.25 nats, but only on labels below `1e-9`
probability, which is bf16 rounding and not a disagreement that can reach a
decision — the same caveat the vLLM parity check reports.

## L2 on images

`fit_head` and `level="L2"` take image states unchanged: the decider renders each state with
its pictures through the processor, and `VLMBackend.hidden_states` returns the language model's
last-position hidden state from the same forward the readout uses (the label log-probs of the
two agree exactly; `tests/test_vlm_engine.py`). Two differences from text:

- **No early stop.** The block loop that stops a text forward at the head's block cannot run a
  vision model, whose positions come from the image grid, so an image decision runs the whole
  forward and its diagnostics carry `early_stop=False`. It is still one forward per state,
  where L0 costs K.
- **A head can memorise pictures.** A temperature has one parameter; a head has thousands. If
  the same picture sits in the labelled set and in the traffic (several questions about one
  image), split by picture before reading a score.

```python
from anyjev import Decider, Image, Question
from anyjev.backends.hf_vlm import VLMBackend

d = Decider(VLMBackend("Qwen/Qwen3-VL-4B-Instruct"))
breed = Question.choice("Which breed is the pet in this photo?", breeds, name="breed")

states = [Image(p) for p in labelled_paths]          # any state with pictures, as for L0
d.calibrate(breed, states, labels)                   # L1: a temperature
d.fit_head(breed, states, labels)                    # L2: one forward per state, then a closed-form solve
d.save_artifacts("qwen3-vl-4b.json")                 # both, one file; d.load_artifacts(...) restores them

r = d.decide(Image("new.jpg"), [breed], level="auto")   # L2 where a head exists, else L1, else L0
r["breed"].level, r["breed"].diagnostics["blocks_executed"], r["breed"].diagnostics["early_stop"]
```

`labels` are option indices, 100–300 per question in practice (the `pets20` numbers use 200).
Measured on `pets20` and `pope`, three models and three split seeds:
[results_multimodal.md](results_multimodal.md#l2-on-images-a-closed-form-head-per-question).

## Cost

A `choice` with K options still costs K prefills per decision, all sharing the
state prefix. The difference is that the prefix now contains the image tokens,
so the prefill is longer: at the processor's default budget a single image is
several hundred tokens, and `VLMBackend(max_pixels=...)` caps it. Images are
decoded once per distinct prompt, not once per permutation — the decider
deduplicates on `(text, image keys)`, so the K rotations of one state reuse
one decoded picture.

## Running the benchmarks and the tests

From a checkout with `pip install -e ".[vlm,bench,dev]"`; every command writes JSON under
`bench/` and the tables in [results_multimodal.md](results_multimodal.md) are printed from it.

```bash
bash bench/run_mm.sh                         # raw / L0 / L1 on pets20, pope, ai2d; 3 models, one per GPU
python -m bench.table bench/results_mm/<date>    # the full table of one results directory

bash bench/run_mm_l2.sh                      # L0 / L1 / L2 on pets20 and pope; 3 models x seeds 0-2
python -m bench.mm_l2_audit bench/results_mm_l2/<date>   # the L2 tables, paired intervals, leakage check

GPUS="2 3" MODELS="4B" SEEDS="0" CKPT=/path/to/checkpoints bash bench/run_mm_l2.sh   # a subset, local weights
```

Both scripts take `GPUS`, `MODELS`, `CKPT` (a local checkpoint root instead of the hub) and
`BS`; logs go to `bench/logs_mm/`. Image preprocessing is CPU bound, so the scripts cap
`OMP_NUM_THREADS`.

```bash
pytest -q                                                    # unit tests, no GPU: the image path on a synthetic backend
ANYJEV_VLM_MODEL=Qwen/Qwen3-VL-2B-Instruct pytest -m engine tests/test_vlm_engine.py   # a real model on a GPU
python scripts/smoke_vlm.py Qwen/Qwen3-VL-2B-Instruct   # colour swatches and the batch-parity check
```
