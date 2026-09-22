# -*- coding: utf-8 -*-
"""Muse-Glimmer-30B's side of the import front door's target contract.

``tools/convert/import_model.py`` does not keep its own copy of any target's
contract.  ``evaluate_targets`` imports ``tools.convert.<target>.convert`` and
calls the ``validate_config`` it finds there; the runnable gate reads
``SOURCE_QUANT_METHODS`` / ``SUPPLIES_FRONTEND_RESOURCES`` from the same module.
This module is Muse's answer to that contract, so the front door's verdict about
Muse is decided by Muse's own numbers rather than by a second copy kept in the
front door.

Why this file exists
--------------------
The front door already listed ``muse_glimmer_30b`` in ``REGISTERED_TARGETS``
(import_model.py:62) and already accepted ``muse_glimmer`` as a decoder family
(import_model.py:82), but the converter exposed no pure-config entry, so
``evaluate_targets`` reported the target as *unavailable* -- measured, verbatim:

    [?] muse_glimmer_30b     该 target 未提供纯 config 校验入口（需带权重的 preflight）

and the run then closed with "4) 该形状没有被任何注册 target 接受".  The front
door could name Muse and still refuse to route it, which is exactly the failure
mode where "recognized" is misread as "routed".

Every number here is either read from the converter module at call time (so the
validator and the object plan cannot disagree) or is a value the engine target
hard-codes and therefore *requires* the checkpoint to agree with
(``src/targets/muse_glimmer_30b/impl/config.h``).
"""
from __future__ import annotations

from typing import Any, Mapping

from tools.convert.qwen3_6.common.conversion import check_members

# --------------------------------------------------------------------------- #
# source flavour / frontend provenance this converter owns
# --------------------------------------------------------------------------- #
#: The NVIDIA NVFP4-QAT Muse checkpoint is a ModelOpt quantisation: its
#: config.json carries ``quantization_config.quant_method == "modelopt"`` with
#: ``quant_algo == "MIXED_PRECISION"`` (measured: 180 fp8 targets + 213 nvfp4
#: targets, and "全部 401 个 Linear" per the front door's flavour census).
#: ``convert.py`` reads those quantised tensors directly -- fp8 row-scale
#: re-encode for the attention projections, NVFP4 pass-through for mlp / lm_head
#: -- so the front door may run it on a ``modelopt`` source.  A target that does
#: not declare this keeps the ``native``-only default it had before this hook
#: existed.
SOURCE_QUANT_METHODS = ("modelopt",)

#: The artifact's six frontend resources are synthesised by this converter from
#: pinned copies under ``tools/convert/muse_glimmer_30b/`` (see
#: ``convert.resource_specs``), not taken from the source's sidecars: the source
#: tokenizer_config is only read to patch a few fields, and tokenizer.json is
#: embedded from the source.  So the source-side six-resource pin comparison is
#: *not* evidence about the artifact, and the front door must not use it as the
#: acceptance criterion for this target.  The real check still happens and is
#: stronger: ``chat_template.cpp:414-424`` refuses any artifact whose
#: ``frontend/chat_template.jinja`` digest is not one of the two compiled-in
#: ones, and the artifact is only accepted after ``verify_artifact`` loads it and
#: it emits at least one token.
SUPPLIES_FRONTEND_RESOURCES = True

# --------------------------------------------------------------------------- #
# checkpoint identity (config.json side; the artifact identity is MODEL_ID /
# WEIGHTS_ID in convert.py and is asserted against the engine below)
# --------------------------------------------------------------------------- #
_ROOT_CONFIG = {
    "architectures": ["MuseGlimmerForConditionalGeneration"],
    "model_type": "muse_glimmer",
    "image_token_id": 200092,
    "video_token_id": 200091,
}

#: Values the engine target compiles in, so a checkpoint that disagrees would
#: convert into an artifact the engine would run with different semantics.
#: Provenance (all in src/targets/muse_glimmer_30b/impl/config.h):
#:   rms_epsilon 1e-05 (:36)              post_norm_eps 1e-08 (:37)
#:   rope_theta 500000.0 (:38)            sliding_window 2048 (:41)
#:   qk_scale_factor 3.87 (:42)           final_logit_softcapping 20.0 (:43)
#:   output_multiplier() 0.19611613513818404 (:79)
#:   kAttentionScale 0.34206290539899237 == qk_scale_factor / sqrt(head_dim)
#:     (3.87 / sqrt(128) = 0.3420629...), i.e. the attention scale is *derived*
#:     from the config's qk_scale_factor, which is why it is pinned here.
_TEXT_SCALARS = {
    "model_type": "muse_glimmer_text",
    "sliding_window": 2048,
    "rms_norm_eps": 1e-05,
    "post_norm_eps": 1e-08,
    "max_position_embeddings": 131072,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "qk_scale_factor": 3.87,
    "output_multiplier": 0.19611613513818404,
    "final_logit_softcapping": 20.0,
}

_ROPE_CONFIG = {"rope_theta": 500000.0, "rope_type": "default"}

#: Vision tower, pinned for the same reason: the converter writes a text-first
#: artifact and the engine's vision path is registered with these dimensions.
_VISION_CONFIG = {
    "model_type": "muse_glimmer_vision",
    "hidden_size": 1536,
    "num_hidden_layers": 50,
    "intermediate_size": 8960,
    "num_attention_heads": 16,
    "patch_size": 14,
    "patch_temporal": 2,
    "merge_size": 2,
}

#: The 52-layer schedule: "full_attention" at 3, 7, 11 ... 51 (every 4th from 3),
#: "sliding_attention" elsewhere -> 13 full + 39 sliding.  The same 13 slots are
#: the NoPE slots in the engine's ``layer_rope_theta`` table (config.h:59-62
#: stores 500000.0 there and 0.0 elsewhere, and config.h:118-124 asserts exactly
#: 13 zeros), so the two tables are one schedule and are checked as one.
_FULL_ATTENTION_EVERY = 4
_FULL_ATTENTION_OFFSET = 3


def _converter() -> Any:
    """The converter module, imported lazily to avoid an import cycle.

    ``convert.py`` imports this module, so this module must not import it at
    module scope.  Importing it here keeps the geometry single-sourced: the
    validator compares the checkpoint against the very constants the object plan
    is built from.
    """
    from . import convert as _convert

    return _convert


def _expected_layer_types(layers: int) -> tuple[str, ...]:
    return tuple(
        "full_attention"
        if layer % _FULL_ATTENTION_EVERY == _FULL_ATTENTION_OFFSET
        else "sliding_attention"
        for layer in range(layers)
    )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the registered shape summary, or raise ValueError naming each mismatch.

    Pure config: it never opens a weight shard, which is what lets the front door
    route on it (import_model.py:409-413 is the caller that needs this entry).
    """

    convert = _converter()

    _geometry = {
        "hidden_size": convert.HIDDEN,
        "num_hidden_layers": convert.LAYERS,
        "intermediate_size": convert.INTERMEDIATE,
        "num_attention_heads": convert.QHEADS,
        "num_key_value_heads": convert.KVHEADS,
        "head_dim": convert.HD,
        "vocab_size": convert.VOCAB,
    }

    check_members("config", config, _ROOT_CONFIG)

    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping):
        raise ValueError("config.json must contain text_config (Muse nests the text tower)")
    if not isinstance(vision, Mapping):
        raise ValueError("config.json must contain vision_config")

    check_members("text_config", text, {**_TEXT_SCALARS, **_geometry})

    expected_layer_types = _expected_layer_types(convert.LAYERS)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            "text_config.layer_types does not match the registered "
            f"{convert.LAYERS}-layer schedule "
            f"(full_attention at {_FULL_ATTENTION_OFFSET}, "
            f"{_FULL_ATTENTION_OFFSET + _FULL_ATTENTION_EVERY}, ... "
            f"{convert.LAYERS - 1})"
        )

    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)

    check_members("vision_config", vision, _VISION_CONFIG)

    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "artifact_identity": {"model_id": convert.MODEL_ID, "weights_id": convert.WEIGHTS_ID},
        "text": {name: text[name] for name in {**_TEXT_SCALARS, **_geometry}},
        "layer_schedule": {
            "layers": convert.LAYERS,
            "full_attention": sum(1 for k in expected_layer_types if k == "full_attention"),
            "sliding_attention": sum(1 for k in expected_layer_types if k == "sliding_attention"),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
    }


__all__ = [
    "SOURCE_QUANT_METHODS",
    "SUPPLIES_FRONTEND_RESOURCES",
    "validate_config",
]
