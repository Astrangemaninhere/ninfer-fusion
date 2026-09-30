# -*- coding: utf-8 -*-
"""Hugging Face source recipe for the Gemma-4-31B inventory.

Object names and shapes are ``inventory``'s; this module is the *other*
direction -- which source keys each object is made of, and what has to be done
to them.  The two are one schema, so ``validate_coverage`` checks them against
each other and against the checkpoint's own key list rather than against a
written-down expectation.

Measured source shape (header of ``data/gemma4-31B/model.safetensors``, 2418
tensors, read without touching the payload):

============================  ===========================================
key                           stored dtype / shape
============================  ===========================================
``<base>.weight_packed``      ``U8``     ``(N, K/2)``
``<base>.weight_scale``       ``F8_E4M3`` ``(N, K/16)``
``<base>.weight_global_scale````F32``    ``(1,)``
``<base>.input_global_scale`` ``F32``    ``(1,)``
============================  ===========================================

410 such ``<base>``s = 60 layers x 6 projections + 50 value projections, which
is the whole of the checkpoint's U8 population: nothing else is quantised.

Same four quantities as ``tools/convert/dequant/modelopt.py`` reads for the
Qwen3.8 NVFP4 checkpoint, under different key names::

    this source                        modelopt reader
    <base>.weight_packed          <->  <base>.weight
    <base>.weight_scale           <->  <base>.weight_scale
    <base>.weight_global_scale    <->  <base>.weight_scale_2 (ROLE match; RECIPROCAL as stored)
    <base>.input_global_scale     <->  <base>.input_scale    (ROLE match; RECIPROCAL as stored)

so ``tools/artifact/layouts.py``'s ``encode_nvfp4`` (which the Qwen3.8 NVFP4
converter already uses, and which its own docstring describes as "Encode exact
source NVFP4 words without numerical conversion") is reusable once a reader
exists that speaks the ``*_global_scale`` spelling.  Writing that reader is the
work item ``convert.py`` names when it refuses to materialise; this module does
not pretend it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import inventory

EMBED_KEY = "model.language_model.embed_tokens.weight"
FINAL_NORM_KEY = "model.language_model.norm.weight"

#: Source subtrees this text-first recipe deliberately does not consume.  Every
#: entry is named rather than pattern-matched away, so ``validate_coverage``
#: can report an unconsumed key that is *not* on this list instead of silently
#: accepting it.
IGNORED_SOURCE_PREFIXES = (
    "model.vision_tower.",
    "model.embed_vision.",
    "model.audio_tower.",
)

_QUANT_FIELDS = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")


@dataclass(frozen=True, slots=True)
class ObjectRecipe:
    object_name: str
    source_keys: tuple[str, ...]
    op: str
    note: str = ""


def _layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}."


def quant_sources(base: str) -> tuple[str, ...]:
    """The four packed/scale keys one quantised linear is stored as."""

    return tuple(f"{base}.{field}" for field in _QUANT_FIELDS)


def _build_recipes() -> tuple[ObjectRecipe, ...]:
    recipes: list[ObjectRecipe] = [
        ObjectRecipe("text/token_embedding", (EMBED_KEY,), "pass-through-bf16",
                     "stored BF16 (measured); not packed"),
        ObjectRecipe("text/output_head", (EMBED_KEY,), "materialize-copy-bf16",
                     "tie_word_embeddings=true and the source has no lm_head tensor, "
                     "so the head is a copy of the embedding "
                     "(gemma_engine_plan.md:56-58)"),
    ]
    for layer in range(inventory.LAYERS):
        prefix = inventory.kind_of(layer)
        src = _layer_prefix(layer)
        obj = f"text/layers/{layer}/"
        attn = src + "self_attn."

        recipes.extend(
            (
                ObjectRecipe(obj + "input_norm",
                             (src + "input_layernorm.weight",), "pass-through-bf16"),
                ObjectRecipe(obj + "attention/query",
                             quant_sources(attn + "q_proj"), "nvfp4-pass-through"),
                ObjectRecipe(obj + "attention/key",
                             quant_sources(attn + "k_proj"), "nvfp4-pass-through"),
                ObjectRecipe(obj + "attention/output",
                             quant_sources(attn + "o_proj"), "nvfp4-pass-through"),
                ObjectRecipe(obj + "attention/query_norm",
                             (attn + "q_norm.weight",), "pass-through-bf16"),
                ObjectRecipe(obj + "attention/key_norm",
                             (attn + "k_norm.weight",), "pass-through-bf16"),
                ObjectRecipe(obj + "post_attention_norm",
                             (src + "post_attention_layernorm.weight",), "pass-through-bf16"),
                ObjectRecipe(obj + "pre_feedforward_norm",
                             (src + "pre_feedforward_layernorm.weight",), "pass-through-bf16"),
                ObjectRecipe(obj + "post_feedforward_norm",
                             (src + "post_feedforward_layernorm.weight",), "pass-through-bf16"),
                ObjectRecipe(obj + "layer_scalar",
                             (src + "layer_scalar",), "pass-through-bf16",
                             f"per-layer scalar; present on all {inventory.LAYERS} layers "
                             f"(measured); its multiply site in the engine is still "
                             f"undecided (gemma_engine_plan.md:95-96)"),
                ObjectRecipe(obj + "mlp/gate",
                             quant_sources(src + "mlp.gate_proj"), "nvfp4-pass-through"),
                ObjectRecipe(obj + "mlp/up",
                             quant_sources(src + "mlp.up_proj"), "nvfp4-pass-through"),
                ObjectRecipe(obj + "mlp/down",
                             quant_sources(src + "mlp.down_proj"), "nvfp4-pass-through"),
            )
        )
        if prefix == "sliding":
            recipes.append(
                ObjectRecipe(obj + "attention/value",
                             quant_sources(attn + "v_proj"), "nvfp4-pass-through")
            )
        else:
            recipes.append(
                ObjectRecipe(obj + "attention/value",
                             (attn + "k_proj.weight_packed",
                              attn + "k_proj.weight_scale",
                              attn + "k_proj.weight_global_scale"), "alias-of-key",
                             "attention_k_eq_v: the ten full layers store no v_proj "
                             "(measured: 410 U8 tensors = 60*6 + 50); value is read off key")
            )

    recipes.extend(
        (
            ObjectRecipe("text/final_norm", (FINAL_NORM_KEY,), "pass-through-bf16"),
        )
    )
    return tuple(recipes)


OBJECT_RECIPES = _build_recipes()

RECIPES_BY_OBJECT = {recipe.object_name: recipe for recipe in OBJECT_RECIPES}


def planned_objects() -> tuple[str, ...]:
    return tuple(recipe.object_name for recipe in OBJECT_RECIPES)


def source_keys() -> tuple[str, ...]:
    seen: list[str] = []
    for recipe in OBJECT_RECIPES:
        for key in recipe.source_keys:
            if key not in seen:
                seen.append(key)
    return tuple(seen)


def validate_coverage(actual_keys) -> dict:
    """Compare the recipe against the checkpoint's own key list.

    Returns a report; it does not raise, because the front door and the
    converter both want to *print* what is wrong rather than catch an
    exception.  ``ok`` is the verdict and the three name lists say which way it
    failed:

    * ``unplanned_source_keys`` -- a source key this recipe does not consume and
      that is not on ``IGNORED_SOURCE_PREFIXES``.  A key nobody planned is a
      silent drop: the object it belonged to is missing from the artifact.
    * ``missing_source_keys`` -- a planned source key the checkpoint does not
      have.  The recipe is written against a different checkpoint.
    * ``object_shape_conflicts`` -- a planned object whose shape disagrees with
      the source key's stored shape (only checked where the shape is
      comparable: the unpacked row count and the K/2 packing).
    """

    actual = tuple(actual_keys)
    actual_set = set(actual)
    planned = source_keys()
    planned_set = set(planned)
    ignored = tuple(
        key for key in actual
        if any(key.startswith(prefix) for prefix in IGNORED_SOURCE_PREFIXES)
    )
    unplanned = tuple(sorted(
        key for key in actual_set - planned_set
        if not any(key.startswith(prefix) for prefix in IGNORED_SOURCE_PREFIXES)
    ))
    missing = tuple(sorted(planned_set - actual_set))

    return {
        "ok": not unplanned and not missing,
        "actual_key_count": len(actual),
        "planned_key_count": len(planned),
        "consumed_key_count": len(planned_set & actual_set),
        "ignored_key_count": len(ignored),
        "ignored_prefixes": IGNORED_SOURCE_PREFIXES,
        "unplanned_source_keys": unplanned,
        "missing_source_keys": missing,
        "planned_object_count": len(OBJECT_RECIPES),
    }


def shape_check(header) -> dict:
    """Check the planned objects' shapes against a safetensors header mapping.

    ``header`` maps source key -> {"dtype": str, "shape": [int, ...]}.  Only the
    comparisons the storage contract actually determines are made: an NVFP4 row
    count must equal the source's row count and its K must be twice the stored
    packed width; a BF16 object must match both dims.
    """

    conflicts: list[str] = []
    checked = 0
    for recipe in OBJECT_RECIPES:
        spec = next((s for s in inventory.TEXT_TENSOR_SPECS
                     if s.name == recipe.object_name), None)
        if spec is None or recipe.op == "materialize-copy-bf16":
            continue
        key = recipe.source_keys[0]
        entry = header.get(key)
        if entry is None:
            continue
        stored = tuple(int(dim) for dim in entry.get("shape", ()))
        if recipe.op == "nvfp4-pass-through":
            if len(stored) != 2 or len(spec.shape) != 2:
                conflicts.append(f"{recipe.object_name}: rank mismatch vs {key} {stored}")
                continue
            if stored[0] != spec.shape[0]:
                conflicts.append(
                    f"{recipe.object_name}: rows {spec.shape[0]} != {key} rows {stored[0]}")
            elif stored[1] * 2 != spec.shape[1]:
                conflicts.append(
                    f"{recipe.object_name}: K {spec.shape[1]} != 2 * {key} packed K "
                    f"{stored[1]} = {stored[1] * 2}")
            checked += 1
        elif recipe.op == "pass-through-bf16":
            if stored != spec.shape:
                conflicts.append(f"{recipe.object_name}: {spec.shape} != {key} {stored}")
            checked += 1
    return {"ok": not conflicts, "checked": checked, "conflicts": tuple(conflicts)}


__all__ = [
    "ObjectRecipe", "OBJECT_RECIPES", "RECIPES_BY_OBJECT", "IGNORED_SOURCE_PREFIXES",
    "EMBED_KEY", "FINAL_NORM_KEY", "quant_sources", "planned_objects", "source_keys",
    "validate_coverage", "shape_check",
]
