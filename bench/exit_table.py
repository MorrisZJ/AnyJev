"""Regenerate docs/results_exit.md from the depth / Jev-mode / latency / artifact JSONs.

    python -m bench.exit_table --date 2026-09-22 > docs/results_exit.md

Reads, per model found under bench/results_exit/<date>/: <model>.jevmode.json (the product
table, depth chosen on calibration data), <model>.depth.json (accuracy versus block), and when
present <model>.latency.json (ms per decision by depth) and <model>.artifact.json (the shipped
heads validated live); then, from bench/results_paraphrase/<date>/, <model>.paraphrase.b<block>.json
and <model>.order.json (the same question reworded or re-listed, with label-free adaptation), and from
bench/results_distill/<date>/, <teacher>.distill_heads.{soft,fit}.json (closed-form distillation).
Nothing is typed by hand: the README's Jev-mode table is the first table printed here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

MODEL_ORDER = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B-Instruct-2507",
               "Qwen/Qwen3-32B"]
PUBLISHED = [("Jev (published)", 0.727),
             ("laya-typed-decisions, fine-tuned on all 300 train cases per question", 0.768)]
NAN = float("nan")


def load(root: str, model: str, kind: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(root, f"{model.replace('/', '__')}.{kind}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct-2507", "")


def cost_ratio(lat: Optional[Dict[str, Any]], block: int, tokens: int = 110) -> Optional[Tuple[float, int]]:
    """(batched ms at the measured block nearest to `block` / raw batched ms, that block)."""
    if lat is None:
        return None
    rows = [r for r in lat["rows"] if r["state_tokens"] == tokens]
    raw = next((r for r in rows if r["mode"].startswith("raw")), None)
    trunc = [r for r in rows if not r["mode"].startswith("raw")]
    if raw is None or not trunc:
        return None
    near = min(trunc, key=lambda r: abs(r["blocks"] - block))
    return near["batch_ms"] / raw["batch_ms"], near["blocks"]


def jev_mode_table(root: str, models: List[str]) -> List[str]:
    out = ["## Jev mode: one closed-form head per question at one block per model", "",
           "LocalLLaMA/typed-decisions, 20 questions, 300 labelled decisions per question to fit, 100 held out per",
           "question to test (2000 pooled). The block is the shallowest whose pooled out-of-fold accuracy on the",
           "calibration data is within 0.5 points of the best (`bench.jev_mode_table`); raw / L0 / L1 are the",
           "regenerated typed study; the cost is the batched ms per decision at the nearest measured block,",
           "relative to one plain forward, for 110- and 1000-token states (`bench.exit_latency`).", "",
           "| model | raw | L0 | L1 | L2, fixed block | 95% CI | vs full-depth head | ECE | flip | block "
           "| ms per decision vs full forward (110 / 1000 tokens) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        jm = load(root, m, "jevmode")
        refs = jm.get("references") or {}
        b = jm["recommended_block"]
        r = jm["rules"][f"fixed@{b}"]
        lat = load(root, m, "latency")
        ratios = [cost_ratio(lat, b, tokens) for tokens in (110, 1000)]
        cost = (f"{ratios[0][0]:.2f}x / {ratios[1][0]:.2f}x (block {ratios[0][1]})"
                if all(ratios) else "")
        ref_cells = " | ".join(f"{refs[k]['acc']:.3f}" if k in refs else "" for k in ("raw", "L0", "L1"))
        ci, d = r["acc_ci95"], r["diff_vs_full_ci95"]
        out.append(f"| {short(m)} | {ref_cells} | **{r['acc']:.3f}** | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                   f"[{d[0]:+.3f}, {d[1]:+.3f}] | {r['ece']:.3f} | {r['flip']:.3f} | "
                   f"{b} of {jm['n_blocks']} ({100 * b / jm['n_blocks']:.0f}%) | {cost} |")
    for name, acc in PUBLISHED:
        out.append(f"| {name} | | | | {acc:.3f} | | | | | | |")
    out += ["", "Rules per model (`cv` = each question picks its block by out-of-fold NLL; "
            "`fixed@b` = one block per model):", ""]
    for m in models:
        jm = load(root, m, "jevmode")
        rules = "; ".join(f"{k}: {v['acc']:.3f} at {v['mean_blocks']:.1f} blocks, ECE {v['ece']:.3f}"
                          for k, v in jm["rules"].items())
        oof = " ".join(f"{b}:{a:.3f}" for b, a in zip(jm["blocks"], jm["oof_acc_by_block"]))
        out.append(f"- **{short(m)}** ({jm['n_blocks']} blocks; JSON `{root}/{m.replace('/', '__')}.jevmode.json`): "
                   f"{rules}. Pooled out-of-fold accuracy by block: {oof}.")
    return out


def depth_tables(root: str, models: List[str]) -> List[str]:
    out = ["## Accuracy versus block", "",
           "Per block: the restricted logit lens (final norm + label rows of `lm_head`, no labels), the same with a",
           "temperature, and the per-question head (LDA / ridge by CV) on the last-position state; pooled over the",
           "20 typed questions, test split (`bench.exit_study`).", ""]
    for m in models:
        dp = load(root, m, "depth")
        if dp is None:
            continue
        out += [f"### {short(m)} (`{root}/{m.replace('/', '__')}.depth.json`)", "",
                "| block | depth | lens acc | lens+T acc | lens+T ECE | head acc | head ECE | head flip |",
                "|---|---|---|---|---|---|---|---|"]
        n = max(int(k) for k in dp["per_layer"])
        for k, rec in sorted(dp["per_layer"].items(), key=lambda kv: int(kv[0])):
            lens, lt, hd = rec.get("lens", {}), rec.get("lens+T", {}), rec.get("head", {})
            out.append(f"| {int(k)} | {100 * int(k) / n:.0f}% | {lens.get('acc', NAN):.3f} | "
                       f"{lt.get('acc', NAN):.3f} | "
                       f"{lt.get('ece', NAN):.3f} | {hd.get('acc', NAN):.3f} | {hd.get('ece', NAN):.3f} | "
                       f"{hd.get('flip', NAN):.3f} |")
        out.append("")
    return out


def latency_tables(root: str, models: List[str]) -> List[str]:
    out = ["## Milliseconds per decision by depth", "",
           "One H100 NVL, bf16, never-seen states; batched = 32 prompts per forward (16 for the 32B), single = one",
           "prompt; the raw row is the model's plain forward (`next_token_logprobs`), the others the truncated block",
           "loop (`hidden_states_to(max_layer=)`). GFLOPs = 2 x parameters per block x prompt tokens x blocks.", ""]
    for m in models:
        lat = load(root, m, "latency")
        if lat is None:
            continue
        out += [f"### {short(m)} (`{root}/{m.replace('/', '__')}.latency.json`)", "",
                "| state tokens | prompt tokens | mode | blocks | batched ms | single ms | GFLOPs |",
                "|---|---|---|---|---|---|---|"]
        for r in lat["rows"]:
            out.append(f"| {r['state_tokens']} | {r['prompt_tokens']} | {r['mode'].split(' (')[0]} | "
                       f"{r['blocks']} | "
                       f"{r['batch_ms']:.1f} | {r['single_ms']:.1f} | {r['gflops']:.0f} |")
        out.append("")
    return out


def artifact_table(root: str, models: List[str]) -> List[str]:
    out = ["## Shipped heads, validated live", "",
           "`scripts/build_heads.py`: `Decider.fit_head` on the calibration states, `decide_batch(level=\"L2\")` on",
           "the test states, one block per model; |dp| is the agreement between the live probabilities and the",
           "same head applied to the study cache's features.", "",
           "| model | block | heads | typed pooled acc | typed pooled ECE | banking20 | newsgroups | injection "
           "| ms per decision (batch 16) | mean abs dp vs cache |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        art = load(root, m, "artifact")
        if art is None:
            continue
        v, s = art["validation"], art["summary"]
        tp = s.get("typed_pooled") or {}
        bench = {k.split(".", 1)[1]: r for k, r in v.items() if k.startswith("bench.")}
        blocks = "/".join(str(b) for b in sorted({r["layer_abs"] for r in v.values()}))
        bench_cells = " | ".join(f"{bench[t]['acc']:.3f}" if t in bench else ""
                                 for t in ("banking20", "newsgroups", "injection"))
        out.append(f"| {short(m)} | {blocks} of {art['n_blocks']} | {len(v)} | {tp.get('acc', NAN):.3f} | "
                   f"{tp.get('ece', NAN):.3f} | {bench_cells} | {s['mean_ms_per_decision']:.1f} | "
                   f"{s['cache_mean_abs_dp']:.4f} |")
    return out


def paraphrase_tables(root: str) -> List[str]:
    out = ["## The same question reworded or re-listed", "",
           "`bench.paraphrase_study`: three hand-written rewordings of each typed question (w1 light, w2",
           "restructured, w3 reframed) and one of its option descriptions (o1). Columns: the head fit on the",
           "original wording applied as is; the same head with its feature mean and scale re-estimated on the",
           "variant's 300 train states without labels (`ta`) or on the 100 test states (`tt`); a head refit on",
           "the variant's own labels; flip = share of test states whose answer differs from the original",
           "wording's. Pooled over the 20 typed questions (2000 test decisions per row).", ""]
    for path in sorted(glob.glob(os.path.join(root, "*.paraphrase.b*.json"))):
        d = json.load(open(path))
        rows = d["rows"]
        out += [f"### {short(d['model'])}, block {d['block']} (`{path}`)", "",
                "| variant | raw | L0 | original head as is | vs original (95% CI) | + recentred (ta) "
                "| + recentred (tt) | own refit | mixed wordings | all wordings | flip vs original "
                "| L0 flip vs original |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in d["variants"]:
            r = rows[f"orig-head@{v}"]
            ci = r["diff_vs_orig_ci95"]

            def c(m: str) -> str:
                return f"{rows[m]['acc']:.3f}" if m in rows else ""

            out.append(f"| {v} | {c(f'raw@{v}')} | {c(f'L0@{v}')} | {r['acc']:.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                       f"{c(f'ta-head@{v}')} | {c(f'tt-head@{v}')} | {c(f'own-head@{v}')} | {c(f'mix-head@{v}')} | "
                       f"{c(f'all-head@{v}')} | {r.get('flip_vs_orig', 0.0):.3f} | "
                       f"{rows[f'L0@{v}'].get('flip_vs_orig', 0.0):.3f} |")
        budgets = sorted({int(m.split("@")[1].split("-")[0]) for m in rows if m.startswith("ta@")})
        if budgets:
            out += ["", "Unlabelled states used for the recentring (pooled over the four rewordings, "
                    "mean of 3 draws):",
                    "", "| n unlabelled | mean + scale | mean only |", "|---|---|---|"]
            vs = [v for v in d["variants"] if v != "orig"]
            for n in budgets:
                a = sum(rows[f"ta@{n}-head@{v}"]["acc"] for v in vs) / len(vs)
                b = sum(rows[f"ta-mean@{n}-head@{v}"]["acc"] for v in vs) / len(vs)
                out.append(f"| {n} | {a:.3f} | {b:.3f} |")
            none = sum(rows[f"orig-head@{v}"]["acc"] for v in vs) / len(vs)
            full = sum(rows[f"ta-head@{v}"]["acc"] for v in vs) / len(vs)
            own = sum(rows[f"own-head@{v}"]["acc"] for v in vs) / len(vs)
            out += [f"| 300 | {full:.3f} | |", f"| 0 (head as is) | {none:.3f} | |",
                    f"| refit with 300 labels | {own:.3f} | |"]
        out.append("")
    orders = sorted(glob.glob(os.path.join(root, "*.order.json")))
    if orders:
        out += ["### Options listed in reverse (replayed from the exit-study caches)", "",
                "| model | block | head fit on | canonical | reversed | reversed + recentring "
                "| flip reversed vs canonical |",
                "|---|---|---|---|---|---|---|"]
        for path in orders:
            d = json.load(open(path))
            for fo, label in (("id", "the canonical order only"), ("rand", "a random order per calibration state")):
                r = {k: d["rows"][f"fit-{fo}@{k}"] for k in ("canonical", "reversed", "reversed+tt")}
                out.append(f"| {short(d['model'])} | {d['block']} | {label} | {r['canonical']['acc']:.3f} | "
                           f"{r['reversed']['acc']:.3f} | {r['reversed+tt']['acc']:.3f} | "
                           f"{r['reversed']['flip_vs_canonical']:.3f} |")
        out.append("")
    return out


def distill_tables(root: str) -> List[str]:
    out: List[str] = []
    for path in sorted(glob.glob(os.path.join(root, "*.distill_heads.soft.json"))):
        d = json.load(open(path))
        out += ["## Closed-form distillation", "",
                f"`bench.distill_heads` (`{path}`): soft targets on the same 300 states, teacher "
                f"{short(d['teacher'])} head at block {d['teacher_block']}, out-of-fold probabilities.", "",
                "| student | gold labels | teacher argmax | teacher soft | gold + teacher soft | teacher |",
                "|---|---|---|---|---|---|"]
        for model, block in d["students"]:
            tag = model.split("/")[-1]
            r = d["rows"][tag]
            cells = " | ".join(f"{r[f'{tag}:{v}']['acc']:.3f}"
                              for v in ("gold", "teacher-argmax", "teacher-soft", "gold+teacher-soft"))
            out.append(f"| {tag} @{block} | {cells} | {r['teacher']['acc']:.3f} |")
        out.append("")
    for path in sorted(glob.glob(os.path.join(root, "*.distill_heads.fit.json"))):
        d = json.load(open(path))
        budgets = d["budgets"]
        out += [f"Synthetic states labelled by the teacher's shipped heads (`{path}`); cells are hard / soft targets:",
                "", "| student | gold 300 | " + " | ".join(f"synthetic {n}" for n in budgets) + " | "
                + " | ".join(f"gold 300 + synthetic {n}" for n in budgets) + " | teacher |",
                "|---|---|" + "---|" * (2 * len(budgets)) + "---|"]
        for model, block in d["students"]:
            tag = model.split("/")[-1]
            r = d["rows"][tag]

            def pair(v: str) -> str:
                return f"{r[f'{tag}:{v}-hard']['acc']:.3f} / {r[f'{tag}:{v}-soft']['acc']:.3f}"

            cells = [pair(f"synth{n}") for n in budgets] + [pair(f"gold300+synth{n}") for n in budgets]
            out.append(f"| {tag} @{block} | {r[f'{tag}:gold300']['acc']:.3f} | " + " | ".join(cells)
                       + f" | {r['teacher']['acc']:.3f} |")
        out.append("")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="bench/results_exit")
    ap.add_argument("--date", required=True)
    ap.add_argument("--paraphrase-root", default="bench/results_paraphrase")
    ap.add_argument("--distill-root", default="bench/results_distill")
    args = ap.parse_args(argv)
    root = os.path.join(args.root, args.date)
    found = sorted({os.path.basename(p).split(".jevmode.json")[0].replace("__", "/")
                    for p in glob.glob(os.path.join(root, "*.jevmode.json"))})
    models = [m for m in MODEL_ORDER if m in found] + [m for m in found if m not in MODEL_ORDER]
    out = [f"# Accuracy versus depth, and the Jev mode of each model ({args.date})", "",
           f"Generated by `python -m bench.exit_table --date {args.date}`; every number comes from the JSON "
           "named in each section.", ""]
    out += jev_mode_table(root, models) + [""] + depth_tables(root, models)
    out += latency_tables(root, models) + artifact_table(root, models)
    para = os.path.join(args.paraphrase_root, args.date)
    if os.path.isdir(para):
        out += [""] + paraphrase_tables(para)
    dist = os.path.join(args.distill_root, args.date)
    if os.path.isdir(dist):
        out += distill_tables(dist)
    print("\n".join(out))


if __name__ == "__main__":
    main()
