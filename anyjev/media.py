"""Images in a state.

`Image` is a lazy reference to one picture -- a path, a URL, a data URI, raw
bytes, or a PIL image. Nothing is decoded until a backend asks for it, and
every reference carries a stable `key`, because the decider sends the same
state through K permutations and must not pay for the same picture K times.

The content-free probe blanks *every* modality, not only the text: a flat grey
image stands in for the picture the same way "N/A" stands in for the state.
Its size is fixed so the probe depends on the question alone and its prior
stays cacheable across states.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional, Tuple

CF_IMAGE_SIZE = (448, 448)
CF_IMAGE_FILL = (127, 127, 127)


class MediaError(ValueError):
    pass


def _pil_module():
    try:
        from PIL import Image as PILImage
    except ImportError as exc:                                   # pragma: no cover
        raise MediaError("images need Pillow: pip install 'anyjev[vlm]'") from exc
    return PILImage


def _is_pil(obj: Any) -> bool:
    try:
        from PIL.Image import Image as PILImageClass
    except ImportError:                                          # pragma: no cover
        return False
    return isinstance(obj, PILImageClass)


class Image:
    """One image in a state.

    Build it from whatever you have; `load()` returns an RGB PIL image and
    `key` is a stable id used to deduplicate prompts. Two references built
    from the same source share a key; two references to byte-identical
    pictures reached by different paths do not, which costs a duplicate
    prefill and nothing else.
    """

    __slots__ = ("kind", "src", "_key", "_loaded")

    def __init__(self, src: Any):
        if isinstance(src, Image):
            self.kind, self.src, self._key, self._loaded = src.kind, src.src, src._key, src._loaded
            return
        self._loaded = None
        if _is_pil(src):
            self.kind, self.src = "pil", src
            self._loaded = src
            fingerprint = f"pil:{src.mode}:{src.size}:".encode() + src.tobytes()
        elif isinstance(src, (bytes, bytearray)):
            self.kind, self.src = "bytes", bytes(src)
            fingerprint = b"bytes:" + self.src
        else:
            text = str(src)
            if text.startswith(("http://", "https://")):
                self.kind = "url"
            elif text.startswith("data:"):
                self.kind = "data"
            else:
                self.kind = "path"
            self.src = text
            fingerprint = f"{self.kind}:{text}".encode()
        self._key = hashlib.sha256(fingerprint).hexdigest()[:16]

    @property
    def key(self) -> str:
        return self._key

    def load(self):
        """Decode to an RGB PIL image, once."""
        if self._loaded is None:
            self._loaded = self._decode()
        if self._loaded.mode != "RGB":
            self._loaded = self._loaded.convert("RGB")
        return self._loaded

    def _decode(self):
        img = self._open()
        img.load()      # read now: a lazily opened path holds a file handle until then
        return img

    def _open(self):
        import io

        PILImage = _pil_module()
        if self.kind == "path":
            return PILImage.open(self.src)
        if self.kind == "bytes":
            return PILImage.open(io.BytesIO(self.src))
        if self.kind == "url":
            import urllib.request

            with urllib.request.urlopen(self.src, timeout=30) as response:
                return PILImage.open(io.BytesIO(response.read()))
        if self.kind == "data":
            import base64

            header, _, payload = self.src.partition(",")
            raw = base64.b64decode(payload) if ";base64" in header else payload.encode()
            return PILImage.open(io.BytesIO(raw))
        raise MediaError(f"cannot decode image of kind {self.kind!r}")     # pragma: no cover

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Image) and other._key == self._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        src = self.src if self.kind in ("path", "url") else f"<{self.kind}>"
        return f"Image({src!r}, key={self._key})"


def is_image_like(obj: Any) -> bool:
    """True for things `as_image` accepts without guessing. Bare strings are
    not included: a path and a sentence look the same to the walker."""
    return isinstance(obj, (Image, bytes, bytearray)) or _is_pil(obj)


def as_image(obj: Any) -> Image:
    return obj if isinstance(obj, Image) else Image(obj)


def blank_image(size: Optional[Tuple[int, int]] = None) -> Image:
    """The content-free stand-in for a picture: flat grey, fixed size."""
    PILImage = _pil_module()
    return Image(PILImage.new("RGB", size or CF_IMAGE_SIZE, CF_IMAGE_FILL))
