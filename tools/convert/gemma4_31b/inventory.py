# -*- coding: utf-8 -*-
"""Persistent-object contract for a Gemma-4-31B artifact.

This module contains only storage roles: the object names a `.ninfer` container
would carry, their shapes and their numeric formats.  Source-checkpoint mapping
lives in the sibling ``recipe``; the front door's ``validate_config`` lives in
the sibling ``convert``.

Every number here is either read off the source checkpoint's own header
(``data/gemma4-31B/model.safetensors``: 2418 tensors, BF16 778 / F32 820 /
F8_E4M3 410 / U8 410) or read off ``tools/archkit/specs/gemma4-31b_spec.json``
(the 60-entry ``layer_types`` schedule).  Nothing is inferred from another
model's layout.

Measured source shape this plan is written against
-------------------------------------------------
* text tower, 60 layers, 50 ``sliding_attention`` + 10 ``full_attention``
  (``layer_types``, spec:20-81; the full slots are 5, 11, 17, ... 59)
* two different attention geometries in one checkpoint::

    sliding (50 layers)   q 32x256 = 8192   k/v 16x256 = 4096   head_dim 256
    full    (10 layers)   q 32x512 = 16384  k   4x512 = 2048   head_dim 512

  (``text_config.global_head_dim`` 512, ``num_global_key_value_heads`` 4;
  measured q_proj packed ``[8192,2688]`` on layer 0 and ``[16384,2688]`` on
  layer 5)
* ``attention_k_eq_v`` is true, so the **ten full layers ship no v_proj at
  all** -- measured: 410 quantised linears = 60x6 + 50 (k/v are one tensor for
  the full layers, and value is read off key)
* every layer carries four BF16 norms (input / post_attention /
  pre_feedforward / post_feedforward) and one BF16 ``layer_scalar`` of shape
  ``[1]``
* qk-norm is present as **weights** (``q_norm``/``k_norm``, 256 on the sliding
  layers and 512 on the full ones) -- unlike Muse there is no synthesised
  all-ones substitute to make
* ``tie_word_embeddings`` is true and there is **no ``lm_head`` tensor** in the
  checkpoint, so the output head is a materialised copy of the embedding
  (the route ``gemma_engine_plan.md:56-58`` already prescribed: materialise a
  copy at load time, zero runtime change)
* no ``mtp_num_layers``, no ``mtp/*`` keys and no ``nextn`` keys: this source
  cannot carry speculative decoding of any kind
* no GDN/linear-attention layer: all 60 layers are softmax
* vision tower present (27 layers, hidden 1152, patch 16) and unquantised --
  it is in the checkpoint's own 192-entry ``quantization_config.ignore`` list

Formats
-------
The quantised tensors are ``nvfp4-pack-quantized`` with group size 16 and
``torch.float8_e4m3fn`` scales -- i.e. exactly the geometry
``tools/artifact/layouts.py`` already encodes for the ``NVFP4`` artifact format
(``blockscale-k16-m128x4-v1``, ``encode_nvfp4``: "Encode exact source NVFP4
words without numerical conversion").  The words are therefore a pass-through,
not a dequantisation, which is why this plan asks for ``NVFP4`` and not for a
``BF16`` artifact: no 20 GB intermediate has to exist at any point.

Proposal status
---------------
The C++ side of this target does not exist yet (see ``convert.py``'s
``CPP_TARGET_STATUS``).  Until it does, these object names are this module's
proposal and the file that owns them is the one that will have to agree with
``src/targets/gemma4_31b/impl/load/bindings.cpp``.  That coupling is the same
one ``_AUTOADAPT.md:100-103`` draws, and it is why the names are declared here
in one place instead of being spread over the recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "gemma4-31b"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "gemma4_31b"

# --------------------------------------------------------------------------- #
# geometry (specs/gemma4-31b_spec.json:10-18, cross-checked against the source
# config.json text_config and the safetensors header)
# --------------------------------------------------------------------------- #
HIDDEN = 5376
LAYERS = 60
INTERMEDIATE = 21504
VOCAB = 262144
MAX_CTX = 262144
RMS_EPS = 1e-6
SLIDING_WINDOW = 1024
TIE_WORD_EMBEDDINGS = True
FINAL_LOGIT_SOFTCAPPING = 30.0
HIDDEN_ACTIVATION = "gelu_pytorch_tanh"

#: sliding_attention geometry: 32 query heads x 16 KV heads x 256
Q_HEADS = 32
KV_HEADS = 16
HEAD_DIM = 256
#: full_attention geometry, a *second* geometry inside the same checkpoint
GLOBAL_KV_HEADS = 4
GLOBAL_HEAD_DIM = 512

#: The 60-entry schedule, verbatim from the spec (full_attention slots).
FULL_ATTENTION_LAYERS = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
SLIDING_ATTENTION_LAYERS = tuple(l for l in range(LAYERS) if l not in FULL_ATTENTION_LAYERS)

assert len(FULL_ATTENTION_LAYERS) == 10
assert len(SLIDING_ATTENTION_LAYERS) == 50

#: Per-kind partial rotary fraction: the full layers rotate 0.25 of their head
#: dim (spec:86 ``partial_rotary_factor: 0.25`` -> 128 of 512) and the sliding
#: layers declare ``rope_type: default`` with the key absent, i.e. full width
#: (spec:90-93).  Two values, and the shared ``TextConfig`` has one
#: ``rotary_dim`` slot -- this is the per-layer rotary table
#: ``gen_full_target.py:78-82`` refuses to emit and is a named engine hook.
ROTARY_DIM_BY_KIND = {"sliding": HEAD_DIM, "full": GLOBAL_HEAD_DIM // 4}
ROPE_THETA_BY_KIND = {"sliding": 10000.0, "full": 1000000.0}

# --------------------------------------------------------------------------- #
# numeric formats, by the names the artifact container registers
# --------------------------------------------------------------------------- #
BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
#: ``tools/convert/qwen3_8_27b/inventory_nvfp4.py:30-34`` is the precedent for
#: extending the family's base FORMAT_NAMES with the NVFP4 / FP8 spellings.
NVFP4 = "NVFP4"
CONTIGUOUS_LAYOUT = "contiguous-le-v1"
NVFP4_LAYOUT = "blockscale-k16-m128x4-v1"

FORMAT_NAMES = (BF16, FP32, I32, NVFP4)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, NVFP4_LAYOUT)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layers: tuple[int, ...] | None = None

    @property
    def layout(self) -> str:
        return NVFP4_LAYOUT if self.format == NVFP4 else CONTIGUOUS_LAYOUT


@dataclass(frozen=True, slots=True)
class AliasSpec:
    """A logical object that is a *view* of another object, not new bytes."""

    name: str
    target: str
    note: str


def kind_of(layer: int) -> str:
    return "full" if layer in FULL_ATTENTION_LAYERS else "sliding"


def q_rows(layer: int) -> int:
    return Q_HEADS * (GLOBAL_HEAD_DIM if kind_of(layer) == "full" else HEAD_DIM)


def kv_rows(layer: int) -> int:
    return (GLOBAL_KV_HEADS * GLOBAL_HEAD_DIM if kind_of(layer) == "full"
            else KV_HEADS * HEAD_DIM)


def head_dim_of(layer: int) -> int:
    return GLOBAL_HEAD_DIM if kind_of(layer) == "full" else HEAD_DIM


def rotary_dim_of(layer: int) -> int:
    return ROTARY_DIM_BY_KIND[kind_of(layer)]


def rope_theta_of(layer: int) -> float:
    return ROPE_THETA_BY_KIND[kind_of(layer)]


def _build_text_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        # Measured: the embedding is stored BF16, not packed -- the source's 410
        # U8 tensors are exactly 60x6 + 50, i.e. the projections and nothing else.
        TensorSpec("text/token_embedding", (VOCAB, HIDDEN), BF16),
    ]
    for layer in range(LAYERS):
        prefix = f"text/layers/{layer}/"
        hd = head_dim_of(layer)
        specs.extend(
            (
                TensorSpec(prefix + "input_norm", (HIDDEN,), BF16),
                TensorSpec(prefix + "attention/query", (q_rows(layer), HIDDEN), NVFP4),
                TensorSpec(prefix + "attention/key", (kv_rows(layer), HIDDEN), NVFP4),
                TensorSpec(prefix + "attention/query_norm", (hd,), BF16),
                TensorSpec(prefix + "attention/key_norm", (hd,), BF16),
                TensorSpec(prefix + "attention/output", (HIDDEN, q_rows(layer)), NVFP4),
                # The full layers' value projection is not stored: they are the
                # ``attention_k_eq_v`` slots, so ``attention/value`` is an alias
                # of ``attention/key`` (see ALIAS_SPECS).
                TensorSpec(prefix + "post_attention_norm", (HIDDEN,), BF16),
                TensorSpec(prefix + "pre_feedforward_norm", (HIDDEN,), BF16),
                TensorSpec(prefix + "post_feedforward_norm", (HIDDEN,), BF16),
                # gemma_engine_plan.md:3 lists this as a construction-line item;
                # measured present as ``layers.N.layer_scalar`` BF16 [1] on all 60.
                # Kept BF16 exactly as stored: casting a trained scalar would be a
                # numeric change this plan has no reason to make.
                TensorSpec(prefix + "layer_scalar", (1,), BF16),
                TensorSpec(prefix + "mlp/gate", (INTERMEDIATE, HIDDEN), NVFP4),
                TensorSpec(prefix + "mlp/up", (INTERMEDIATE, HIDDEN), NVFP4),
                TensorSpec(prefix + "mlp/down", (HIDDEN, INTERMEDIATE), NVFP4),
            )
        )
        if kind_of(layer) == "sliding":
            specs.append(
                TensorSpec(prefix + "attention/value", (kv_rows(layer), HIDDEN), NVFP4)
            )
    specs.extend(
        (
            TensorSpec("text/final_norm", (HIDDEN,), BF16),
            # tie_word_embeddings is true and there is no lm_head tensor in the
            # source; the head is materialised from the embedding (a copy, so
            # the runtime needs no transpose path).  BF16 for the same measured
            # reason as the embedding itself.
            TensorSpec("text/output_head", (VOCAB, HIDDEN), BF16),
        )
    )
    return tuple(specs)


TEXT_TENSOR_SPECS = _build_text_specs()

#: Frontend resources, the same six names every registered target uses
#: (``tools/convert/qwen3_6/common/inventory.py`` RESOURCE_SPECS).
FRONTEND_RESOURCE_NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
    "frontend/preprocessor_config.json",
    "frontend/video_preprocessor_config.json",
)

#: Objects that are views, not new bytes.
ALIAS_SPECS = (
    AliasSpec("text/layers/{l}/attention/value", "text/layers/{l}/attention/key",
              "attention_k_eq_v: the 10 full layers store no v_proj (measured)"),
    AliasSpec("text/output_head", "text/token_embedding",
              "tie_word_embeddings=true, no lm_head tensor in the source"),
)

VISION_STATUS = (
    "vision tower present (27 layers, hidden 1152, patch 16) and excluded from "
    "quantisation by the checkpoint's own ignore list; this converter's inventory "
    "is text-first and does not plan vision objects.  A vision-capable target "
    "would add them (tools/archkit/out/gemma4_31b has no vision section either)."
)


def tensor_specs() -> tuple[TensorSpec, ...]:
    return TEXT_TENSOR_SPECS


def format_counts() -> dict[str, int]:
    return {fmt: sum(s.format == fmt for s in TEXT_TENSOR_SPECS) for fmt in FORMAT_NAMES}


def layout_counts() -> dict[str, int]:
    return {lay: sum(s.layout == lay for s in TEXT_TENSOR_SPECS) for lay in LAYOUT_NAMES}


def text_tensor_span_bytes() -> int:
    """Payload bytes the planned text objects would occupy (NVFP4 = K/2 words)."""

    total = 0
    for spec in TEXT_TENSOR_SPECS:
        elements = 1
        for dim in spec.shape:
            elements *= dim
        if spec.format == NVFP4:
            total += elements // 2
            # 16-element blocks, one F8_E4M3 scale each
            total += (elements // 16) if len(spec.shape) == 2 else 0
        elif spec.format == BF16:
            total += elements * 2
        elif spec.format == FP32:
            total += elements * 4
        else:
            total += elements * 4
    return total


def nmt_objects() -> tuple[str, ...]:
    """Named pieces this plan deliberately does not carry, with the reason."""

    return (
        "mtp/*            : the source declares no mtp_num_layers and ships 0 mtp/nextn keys",
        "text/draft_head* : no shortlist head exists in this checkpoint",
        "gdn/*            : all 60 layers are softmax; the family's gdn path is unused",
        "vision/*         : see VISION_STATUS",
    )


__all__ = [
    "MODEL_ID", "WEIGHTS_ID", "TARGET_KEY",
    "HIDDEN", "LAYERS", "INTERMEDIATE", "VOCAB", "MAX_CTX", "RMS_EPS",
    "SLIDING_WINDOW", "TIE_WORD_EMBEDDINGS", "FINAL_LOGIT_SOFTCAPPING",
    "HIDDEN_ACTIVATION", "Q_HEADS", "KV_HEADS", "HEAD_DIM", "GLOBAL_KV_HEADS",
    "GLOBAL_HEAD_DIM", "FULL_ATTENTION_LAYERS", "SLIDING_ATTENTION_LAYERS",
    "ROTARY_DIM_BY_KIND", "ROPE_THETA_BY_KIND",
    "BF16", "FP32", "I32", "NVFP4", "CONTIGUOUS_LAYOUT", "NVFP4_LAYOUT",
    "FORMAT_NAMES", "LAYOUT_NAMES", "TensorSpec", "AliasSpec",
    "FRONTEND_RESOURCE_NAMES", "TEXT_TENSOR_SPECS", "ALIAS_SPECS", "VISION_STATUS",
    "kind_of", "q_rows", "kv_rows", "head_dim_of", "rotary_dim_of", "rope_theta_of",
    "tensor_specs", "format_counts", "layout_counts", "text_tensor_span_bytes",
    "nmt_objects",
]
