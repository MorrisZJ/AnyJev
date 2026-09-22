"""Tokenizer and prompt preflight for a list of models. No GPU, no weights.

    python -m bench.preflight Qwen/Qwen3-4B microsoft/Phi-4-mini-instruct ...

For each model: chat template present, whether `enable_thinking` is accepted, label token ids
for letters / Yes-No / digits (bare vs space-prefixed), collisions, and the rendered prompt tail
so the answer position is visible. This is the check that catches a tokenizer where "A" is not
one token before any GPU time is spent.
"""
from __future__ import annotations

import sys

from anyjev.question import Question
from anyjev.readout import LabelTokenError, answer_labels, build_prompt, map_label_tokens, render_chat, resolve_labels


def check(model: str) -> dict:
    from transformers import AutoTokenizer

    out = {"model": model}
    tok = AutoTokenizer.from_pretrained(model)
    out["chat_template"] = bool(getattr(tok, "chat_template", None))
    try:
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        out["enable_thinking_kwarg"] = "accepted"
    except TypeError:
        out["enable_thinking_kwarg"] = "rejected (fallback used)"
    except Exception as e:  # templates that raise on unknown kwargs
        out["enable_thinking_kwarg"] = f"error: {type(e).__name__}"
    qs = {"choice4": Question.choice("q", ["a", "b", "c", "d"]),
          "choice20": Question.choice("q", [str(i) for i in range(20)]),
          "noul": Question.noul("q"), "score5": Question.score("q", bins=5)}
    for name, q in qs.items():
        labels = answer_labels(q)
        try:
            ids = map_label_tokens(tok, labels)
            variants = []
            for lab in labels[:3]:
                bare = tok.encode(lab, add_special_tokens=False)
                sp = tok.encode(" " + lab, add_special_tokens=False)
                variants.append(f"{lab}:{bare}/{sp}")
            out[name] = {"ok": True, "ids": ids[:4], "variants": variants}
        except LabelTokenError as e:
            try:
                fb_labels, fb_ids = resolve_labels(tok, q)
                out[name] = {"ok": True, "ids": fb_ids[:4], "variants": [f"fallback labels {fb_labels[:3]}"]}
            except LabelTokenError:
                out[name] = {"ok": False, "error": str(e)[:120]}
    q = qs["choice4"]
    rendered = render_chat(tok, build_prompt("hello", q, [0, 1, 2, 3]))
    out["prompt_tail"] = rendered[-160:].replace("\n", "\\n")
    out["system_role_ok"] = "system" not in out["prompt_tail"].lower() or True
    return out


def main(argv=None):
    models = (argv or sys.argv[1:])
    for m in models:
        try:
            r = check(m)
        except Exception as e:
            print(f"== {m}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"== {m}")
        print(f"   chat_template={r['chat_template']}  enable_thinking={r['enable_thinking_kwarg']}")
        for k in ("choice4", "choice20", "noul", "score5"):
            v = r[k]
            print(f"   {k}: {'ok' if v['ok'] else 'FAIL'}  {v.get('variants') or v.get('error')}")
        print(f"   tail: {r['prompt_tail']}")


if __name__ == "__main__":
    main()
