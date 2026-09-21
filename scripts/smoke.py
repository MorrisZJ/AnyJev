"""Quick real-model smoke: the README example on one small model.
    python scripts_smoke.py Qwen/Qwen3-8B
"""
import json
import sys

from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
d = Decider(HFBackend(model))
qs = [
    Question.choice("Which handler should process this request?",
                    ["billing", "technical", "sales", "other"], name="route"),
    Question.noul("Is the proposed tool call destructive or irreversible?", name="safe"),
    Question.score("How complete is the task on a 0 to 1 scale?", bins=5, name="done"),
]
state = {"conversation": [{"role": "user", "content": "My card was charged twice for one order."}],
         "proposed_tool_call": {"name": "refund_charge", "args": {"charge_id": "ch_123"}}}
for level in ("raw", "L0"):
    r = d.decide(state, qs, level=level)
    print(level, json.dumps(r.to_dict(), indent=1))
    for dec in r:
        print("  ", dec.question.id, "answer_mass=%.3f" % dec.diagnostics["answer_mass"],
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dec.diagnostics.items()
               if k.startswith("order_flip")})
