"""Backend protocol. A backend does exactly one thing: given rendered prompts,
return the next-token log-probabilities of the requested token ids.
Everything else (debiasing, calibration, abstention) lives above it."""
from __future__ import annotations

from typing import Any, List, Optional, Protocol, Sequence

import numpy as np


class Backend(Protocol):
    tokenizer: Any   # must provide .encode(text, add_special_tokens=False) and optionally .chat_template
    name: str

    # Optional, multimodal backends only:
    #   accepts_images: bool  -- True if next_token_logprobs takes `images`
    #   chat_renderer         -- holds the chat template when it is not the
    #                            tokenizer (a processor, for vision models)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]],
                            images: Optional[Sequence[Sequence[Any]]] = None) -> List[np.ndarray]:
        """For prompt i, return log p(token | prompt_i) for each id in token_ids[i],
        taken from the full-vocabulary log-softmax at the last position.

        `images[i]` is the ordered list of `anyjev.media.Image` for prompt i,
        one per image placeholder the chat template rendered. Text-only
        backends may omit the parameter; the decider only passes it when a
        state actually carries a picture.

        Decision mode never samples; a backend that cannot expose restricted
        next-token logits is not a supported backend."""
        ...
