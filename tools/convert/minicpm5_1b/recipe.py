# -*- coding: utf-8 -*-
"""minicpm5_1b.recipe — the EXPRESSION half: role -> payload, plus tie decisions.

Two roles only, and both are one expression:

  ``matrix``  the artifact stores exactly what the shard stores, so the payload is
              a byte RANGE of the shard (`source.raw_range`).  Nothing is
              dequantized, cast or multiplied -- that is why this converter runs on
              the CPU with no GPU and, for the pass-through path, without torch.
  ``norm``    RMSNorm weights are stored as `w`, and the family runtime applies
              `(1 + w)` (the convention `muse_glimmer_30b/convert.py:bake_norm`
              carries as `1+w` baking), so the sum is materialised here ONCE at
              conversion time instead of once per layer per token.

TIE DECISIONS are the other thing this file owns.  A tie is not a payload: it is
an ALIAS, so the driver writes the same bytes under two names and pays for them
once.  `tie_word_embeddings=true` in the checkpoint therefore changes the
DECLARATION (`ObjectDecl.tie_to`) and not a single byte of the recipe table --
which is what makes it a decision rather than a special case.

Both functions take `(obj, source, ctx)`: the driver calls them and never inspects
what they return beyond "it is a buffer or an iterable of buffers".
"""

from __future__ import annotations

from typing import Any, Iterator

from tools.archkit.kit.contract import KitRefusal, ObjectDecl

#: the tie rule, as one function so that "what does a tie mean here" has one home
TIE_TARGET = "text/token_embedding"


def tie_of(config: dict, object_name: str) -> str:
    """Return the object `object_name` aliases, or "" when it is not tied."""
    if object_name != "text/output_head":
        return ""
    if bool(config.get("tie_word_embeddings", False)):
        return TIE_TARGET
    return ""


def matrix(obj: ObjectDecl, source, ctx) -> Iterator[memoryview]:
    """Pass-through: the payload is the source tensor's own bytes, streamed."""
    if len(obj.sources) != 1:
        raise KitRefusal(
            "role 'matrix' takes exactly one source key; object %s declares %d"
            % (obj.name, len(obj.sources)))
    yield from source.raw_range(obj.sources[0])


def norm(obj: ObjectDecl, source, ctx) -> bytes:
    """RMSNorm `w` baked to `1 + w`, BF16 in BF16 out."""
    import torch  # local: the pass-through path must stay importable without torch

    from tools.artifact.layouts import encode_direct

    key = obj.sources[0]
    weight = torch.frombuffer(bytearray(source.raw_bytes(key)),
                              dtype=torch.bfloat16).reshape(obj.shape)
    baked = (weight.float() + 1.0).to(torch.bfloat16).contiguous()
    return encode_direct(baked, obj.format)


RECIPES = {
    "matrix": matrix,
    "norm": norm,
}
