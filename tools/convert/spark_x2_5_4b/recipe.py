# -*- coding: utf-8 -*-
"""Source expressions for a Spark-X2.5-4B artifact, plus the two gates.

Three things live here, in the order the artifact is built:

1. **Roles.**  One `ObjectRecipe` per engine object, each naming the source keys it
   consumes.  Spark is a plain decoder -- no view alias, no expert bank, no vision
   tower -- so `inventory.LAYER_OBJECTS` / `LAYER_OBJECT_SOURCES` are the only roles
   and they do not map one-to-one: every layer's eight source leaves become TEN
   objects, because `self_attn.q_k_v_proj.weight` is one fused `[6144, 2560]` tensor
   that the engine binds as three (`bindings.cpp:94-108`, and the split order is the
   reference implementation's own, `modeling_spark.py:170-173`).

   The names are the ENGINE's, and that is a re-pointing with a history: the first
   version of this module invented dot-separated role stems of its own
   (`text/layers.N.fused_qkv`, `mlp_gate`, `attn_output_gate`), and
   `dl/sparktarget/REPORT.md` section 4 tabulated all eight as mismatches against the
   names the target binds.  They are the same names now.
2. **The tie decision.**  `model.safetensors.index.json` carries no `lm_head.weight`
   and `config.json` declares `tie_word_embeddings: true`, so the output head has to
   be materialised at conversion time.  `tie_decision` is the whole rule, and it
   reports the orientation it chose and the byte count it implies instead of
   assuming them.
3. **Two gates over the source**: `validate_coverage` (bidirectional -- every planned
   key exists, and every existing key is planned or ignored by a named prefix) and
   `shape_check` (every planned key's shape in the shard header equals the shape
   `inventory` derived from the declared geometry).

Neither gate reads weights: both work off the index and the safetensors header.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import inventory as inv

#: Source keys that are deliberately not planned, by prefix.  Spark has no vision
#: tower and no MTP head, so this is empty -- and an empty tuple is the strong
#: statement: any source key that is not planned is reported as unplanned, never
#: silently dropped.
IGNORED_SOURCE_PREFIXES: tuple[str, ...] = ()

#: The role the tied head is materialised into.  Named after the engine's own object
#: for a text output head (src/targets/*/impl/load/bindings.cpp bind `text/output_head`
#: to a `[vocab, hidden]` tensor and the runtime reads it independently of the
#: embedding), so the name is the engine's, not this module's invention.
OUTPUT_HEAD_ROLE = "text/output_head"
EMBEDDING_ROLE = "text/token_embedding"
FINAL_NORM_ROLE = "text/final_norm"

#: source leaf -> the ENGINE object leaf it fills.  The right-hand side is the
#: engine's spelling verbatim (`bindings.cpp:136-169`), and the fused q/k/v entry says
#: "three objects" in words because it is the one source leaf that does not fill one
#: object; `inventory.LAYER_OBJECT_SOURCES` carries the actual row boundaries.
LEAF_ROLES = {
    "input_layernorm.weight": "input_norm",
    "post_attention_layernorm.weight": "post_attention_layernorm",
    "mlp.gate_proj.weight": "mlp/gate",
    "mlp.up_proj.weight": "mlp/up",
    "mlp.down_proj.weight": "mlp/down",
    "self_attn.q_k_v_proj.weight": "attention/query + attention/key + attention/value",
    "self_attn.g_proj.weight": "attention/gate",
    "self_attn.out_proj.weight": "attention/output",
}


@dataclass(frozen=True, slots=True)
class ObjectRecipe:
    """One object and the source expression that fills it."""

    role: str
    shape: tuple[int, ...]
    sources: tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(f"object {self.role!r} has no source expression")


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """The resolved source of the tied head, with the reason it was chosen."""

    materialized: bool
    head_in_index: bool
    source_key: str | None
    provenance: str
    transpose: bool
    orientation_verified: bool
    source_shape: tuple[int, ...] | None
    source_bytes: int | None
    reason: str
    tie_flag_conflict: bool = False

    def to_report(self) -> dict:
        return {
            "role": OUTPUT_HEAD_ROLE,
            "materialized": self.materialized,
            "head_in_index": self.head_in_index,
            "materialized_from": self.source_key,
            "transpose": self.transpose,
            "orientation_verified": self.orientation_verified,
            "provenance": self.provenance,
            "source_shape": None if self.source_shape is None else list(self.source_shape),
            "source_bytes": self.source_bytes,
            "reason": self.reason,
            "tie_flag_conflict": self.tie_flag_conflict,
        }


def _source_note(source_key: str, first_row: int | None, last_row: int | None) -> str:
    if first_row is None and last_row is None:
        return ""
    return ("rows [{}, {}) of {} -- the fused q/k/v split, measured from "
            "models/Spark-X2.5-4B/modeling_spark.py:170-173".format(
                first_row, last_row, source_key.rsplit(".", 1)[-1]))


def _build_recipes() -> tuple[ObjectRecipe, ...]:
    """Every object the engine binds, in the target's own bind order.

    The order is `bind_artifact()`'s (`bindings.cpp:190-218`): the embedding first,
    then the layer stack, then `text/final_norm`, then the materialised head.
    """

    out = [
        ObjectRecipe(EMBEDDING_ROLE, inv.EMBEDDING_SHAPE, (inv.EMBED_KEY,),
                     "the checkpoint's one embedding tensor, already [vocab, hidden]"),
    ]
    for layer in range(inv.LAYERS):
        for name, shape, key, first_row, last_row in inv.layer_objects(layer):
            note = _source_note(key, first_row, last_row)
            if name.endswith("attention/gate"):
                note = ("per-head output gate: (q_heads, hidden) -- one scalar per head, "
                        "not a per-channel projection.  The engine's sigmoid_mul "
                        "headwise_gate_shape branch is the consumer (manifest gap "
                        "'attn:headwise_output_gate(sigmoid)')")
            elif name.endswith("attention/output"):
                note = ("[hidden, q_heads*head_dim] -- the contraction dimension is the "
                        "query width, not hidden")
            out.append(ObjectRecipe(name, shape, (key,), note))
    out.append(ObjectRecipe(FINAL_NORM_ROLE, inv.FINAL_NORM_SHAPE, (inv.FINAL_NORM_KEY,),
                            "the stack's closing rmsnorm"))
    out.append(ObjectRecipe(OUTPUT_HEAD_ROLE, inv.EMBEDDING_SHAPE, (inv.EMBED_KEY,),
                            "materialised from the embedding: tie_word_embeddings is "
                            "true and the index ships no lm_head.weight.  The "
                            "orientation comes from tie_decision() and is reported, "
                            "not assumed"))
    return tuple(out)


OBJECT_RECIPES = _build_recipes()
RECIPES_BY_ROLE = {recipe.role: recipe for recipe in OBJECT_RECIPES}


def planned_roles() -> tuple[str, ...]:
    return tuple(recipe.role for recipe in OBJECT_RECIPES)


def source_keys() -> tuple[str, ...]:
    seen: list[str] = []
    for recipe in OBJECT_RECIPES:
        for key in recipe.sources:
            if key not in seen:
                seen.append(key)
    return tuple(seen)


# --------------------------------------------------------------------------- #
# the tie decision
# --------------------------------------------------------------------------- #
def tie_decision(index_keys, config, header=None) -> SourceBinding:
    """Resolve the output head from the index, the config flag, and (optionally) a header.

    The rule, in full:

    * the index has `lm_head.weight`  -> read it; nothing is materialised;
    * the index has no `lm_head.weight` -> materialise `text/output_head` from the
      embedding.  The engine loads an independent `[vocab, hidden]` head object either
      way (src/targets/qwen3_6_27b/impl/load/bindings.cpp binds `text/output_head`
      separately from `text/token_embedding`), so no new operator is needed;
    * orientation: identity when the embedding is already `[vocab, hidden]`, transposed
      when it is `[hidden, vocab]`, and an error when it is neither.  The verdict is
      recorded, not assumed.

    `header` is the embedding tensor's header entry (a dict with `shape`), or None when
    the shard is not on disk: then orientation is reported as UNVERIFIED rather than
    guessed.
    """

    keys = set(index_keys)
    head_names = [name for name in ("lm_head.weight", "model.lm_head.weight") if name in keys]
    tied = config.get("tie_word_embeddings")
    head_in_index = bool(head_names)
    conflict = (tied is False) and not head_in_index
    if head_in_index:
        return SourceBinding(
            materialized=False, head_in_index=True, source_key=head_names[0],
            provenance="index", transpose=False,
            orientation_verified=header is not None,
            source_shape=None if header is None else tuple(header["shape"]),
            source_bytes=None if header is None else _bytes_of(header),
            reason="index ships an output head; nothing to materialise",
            tie_flag_conflict=False)
    if inv.EMBED_KEY not in keys:
        raise ValueError(
            f"no embedding tensor to materialise the head from: looked for "
            f"{inv.EMBED_KEY!r} in a {len(keys)}-key index and did not find it")
    shape = None if header is None else tuple(header["shape"])
    if shape is None:
        transpose, verified = False, False
        reason = ("index has no lm_head.weight and the embedding header is not on disk, "
                  "so the orientation is UNVERIFIED")
    elif shape == inv.EMBEDDING_SHAPE:
        transpose, verified = False, True
        reason = ("index has no lm_head.weight; the embedding header is already "
                  f"{shape} == the head object shape, so the orientation is identity")
    elif shape == tuple(reversed(inv.EMBEDDING_SHAPE)):
        transpose, verified = True, True
        reason = ("index has no lm_head.weight; the embedding header is "
                  f"{shape}, so the head object needs a transpose")
    else:
        raise ValueError(
            f"embedding header {shape} is neither {inv.EMBEDDING_SHAPE} nor its reverse, "
            f"so it cannot serve as the [vocab, hidden] output head")
    return SourceBinding(
        materialized=True, head_in_index=False, source_key=inv.EMBED_KEY,
        provenance="identity-from-embedding" if not transpose else "transposed-from-embedding",
        transpose=transpose, orientation_verified=verified, source_shape=shape,
        source_bytes=None if shape is None else _bytes_of({"shape": list(shape)}),
        reason=reason, tie_flag_conflict=conflict)


def _bytes_of(header: dict) -> int:
    return inv.tensor_bytes(tuple(header["shape"]))


# --------------------------------------------------------------------------- #
# gate 1: bidirectional coverage
# --------------------------------------------------------------------------- #
def validate_coverage(actual_keys) -> dict:
    """Every planned key must exist, and every existing key must be planned or ignored."""

    actual = set(actual_keys)
    planned = set(source_keys())
    missing = sorted(planned - actual)
    ignored = tuple(p for p in IGNORED_SOURCE_PREFIXES if any(k.startswith(p) for k in actual))
    unplanned = sorted(k for k in actual - planned
                       if not any(k.startswith(p) for p in IGNORED_SOURCE_PREFIXES))
    ignored_count = sum(1 for k in actual - planned
                        if any(k.startswith(p) for p in IGNORED_SOURCE_PREFIXES))
    return {
        "ok": not missing and not unplanned,
        "planned_key_count": len(planned),
        "consumed_key_count": len(planned & actual),
        "actual_key_count": len(actual),
        "missing_source_keys": missing,
        "unplanned_source_keys": unplanned,
        "ignored_prefixes": list(ignored),
        "ignored_key_count": ignored_count,
    }


# --------------------------------------------------------------------------- #
# gate 2: shape gate
# --------------------------------------------------------------------------- #
def shape_check(header: dict) -> dict:
    """Compare every planned key's header shape against the derived shape."""

    expected = inv.expected_shapes()
    conflicts: list[str] = []
    checked = 0
    present = 0
    for key, want in expected.items():
        entry = header.get(key)
        if entry is None:
            conflicts.append(f"{key}: absent from the shard header")
            continue
        present += 1
        got = tuple(entry["shape"])
        if got != want:
            conflicts.append(f"{key}: header {got} != derived {want}")
            continue
        checked += 1
    extra = sorted(set(header) - set(expected))
    for key in extra[:10]:
        conflicts.append(f"{key}: present in the header but not planned")
    return {"ok": not conflicts, "checked": checked, "present": present,
            "expected": len(expected), "conflicts": conflicts}
