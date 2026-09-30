# -*- coding: utf-8 -*-
"""minicpm5_1b.inventory — the ROLES half: which object, which source, which role.

This file answers exactly two questions and is not allowed to grow:

  * what is the ordered vocabulary of object NAMES this target emits, and
  * which source tensor(s) each name is built from, under which ROLE.

It does NOT decide payload bytes (that is `recipe.py`), and it does NOT write
anything (that is `tools/archkit/kit/driver.py`).  The split is the whole point:
the name vocabulary is the interface a hand-written C++ target binds to, so it has
to be readable as a list, and it is the same list `impl/load/bindings.cpp` will
hold -- two independent statements of one vocabulary that must not drift, which is
what `check_bindings.py`-style interlocking is for.

Naming follows the family vocabulary already in the tree
(`src/targets/spark_x2_5_4b/impl/load/bindings.cpp:98-198` declares
`text/token_embedding`, `text/layers/N/attention/query`, `text/layers/N/mlp/gate`,
`text/final_norm`, `text/output_head`), because a second spelling of the same
concept is a defect in waiting.
"""

from __future__ import annotations

from typing import Iterator

MODEL_ID = "minicpm5-1b"
WEIGHTS_ID = "bf16"

#: The source prefix every layer tensor carries.  Used for the coverage reading,
#: and it is the ONLY place the checkpoint's own naming is written down.
LAYER_PREFIX = "model.layers."

#: Source keys this target deliberately does not consume.  Empty, and empty is a
#: claim the gate checks: `gate_source_coverage` refuses an ignore rule that
#: matches no index key, so a stale entry cannot survive here.
IGNORED_SOURCE_PREFIXES: tuple[str, ...] = ()

#: (object name suffix, source key suffix, role).  Order inside a layer is the
#: order the artifact directory carries.
LAYER_OBJECTS: tuple[tuple[str, str, str], ...] = (
    ("input_norm",                  "input_layernorm.weight",            "norm"),
    ("attention/query",             "self_attn.q_proj.weight",           "matrix"),
    ("attention/key",               "self_attn.k_proj.weight",           "matrix"),
    ("attention/value",             "self_attn.v_proj.weight",           "matrix"),
    ("attention/output",            "self_attn.o_proj.weight",           "matrix"),
    ("post_attention_layernorm",    "post_attention_layernorm.weight",   "norm"),
    ("mlp/gate",                    "mlp.gate_proj.weight",              "matrix"),
    ("mlp/up",                      "mlp.up_proj.weight",                "matrix"),
    ("mlp/down",                    "mlp.down_proj.weight",              "matrix"),
)

#: The two stack-level objects, in directory order: first, then last.
EMBEDDING_OBJECT = ("text/token_embedding", "model.embed_tokens.weight", "matrix")
OUTPUT_HEAD_OBJECT = ("text/output_head", "lm_head.weight", "matrix")
FINAL_NORM_OBJECT = ("text/final_norm", "model.norm.weight", "norm")


def layer_object_name(layer: int, suffix: str) -> str:
    return "text/layers/%d/%s" % (layer, suffix)


def layer_source_key(layer: int, suffix: str) -> str:
    return "%s%d.%s" % (LAYER_PREFIX, layer, suffix)


def objects(layers: int) -> Iterator[tuple[str, tuple[str, ...], str]]:
    """Yield (artifact object name, source keys, role) in directory order."""
    name, key, role = EMBEDDING_OBJECT
    yield name, (key,), role
    for layer in range(layers):
        for suffix, source_suffix, role in LAYER_OBJECTS:
            yield (layer_object_name(layer, suffix),
                   (layer_source_key(layer, source_suffix),),
                   role)
    name, key, role = FINAL_NORM_OBJECT
    yield name, (key,), role
    name, key, role = OUTPUT_HEAD_OBJECT
    yield name, (key,), role


def shape_of(geometry: dict, name: str) -> tuple[int, ...]:
    """The declared shape of one object name, from geometry alone.

    This is a second, independent statement of the same numbers `recipe.py`
    produces payloads from: if the two disagree the `geometry` gate compares both
    against the shard header, and the header wins.
    """
    hidden = int(geometry["hidden"])
    layers = int(geometry["layers"])
    vocab = int(geometry["vocab"])
    intermediate = int(geometry["intermediate"])
    query = int(geometry["query_heads"]) * int(geometry["head_dim"])
    kv = int(geometry["kv_heads"]) * int(geometry["head_dim"])
    if name == "text/token_embedding":
        return (vocab, hidden)
    if name == "text/output_head":
        return (vocab, hidden)
    if name == "text/final_norm":
        return (hidden,)
    leaf = name.split("text/layers/", 1)[1]
    _index, suffix = leaf.split("/", 1)
    if suffix == "input_norm" or suffix == "post_attention_layernorm":
        return (hidden,)
    if suffix == "attention/query":
        return (query, hidden)
    if suffix == "attention/key" or suffix == "attention/value":
        return (kv, hidden)
    if suffix == "attention/output":
        return (hidden, query)
    if suffix == "mlp/gate" or suffix == "mlp/up":
        return (intermediate, hidden)
    if suffix == "mlp/down":
        return (hidden, intermediate)
    raise KeyError("no shape declared for object %s" % name)
