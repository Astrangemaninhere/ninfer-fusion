# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
"""Closed persistent-object contract for Qwen3.8-Flash-Next.

The source checkpoint deliberately has a much larger tensor directory than the
runtime artifact.  Routed experts are stored as two expert-major banks per
layer, and projection rows consumed by one execution leaf are fused here.
"""

from __future__ import annotations

import os

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    I64,
    RESOURCE_SPECS,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
)


MODEL_ID = "qwen3.8-flash-next"
WEIGHTS_ID = "mixed-nvfp4-fp8-ple-int4"
TARGET_KEY = "qwen3_8_flash_next"

NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_F32S"
PLE_U4 = "U4Z8G16_F16S"
FP8_LAYOUT = "row-scale-f32-v1"
EXPERT_NVFP4_LAYOUT = "expert-blockscale-k16-m128x4-v1"
PLE_U4_LAYOUT = "packed-u4-g16-v1"

FULL_ATTENTION_LAYERS = tuple(range(3, 48, 4))
GDN_LAYERS = tuple(layer for layer in range(48) if layer not in FULL_ATTENTION_LAYERS)


def tensor_spec(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    if numeric_format in (BF16, I64):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format == FP8:
        layout = FP8_LAYOUT
    elif numeric_format == NVFP4:
        layout = EXPERT_NVFP4_LAYOUT
    elif numeric_format == PLE_U4:
        layout = PLE_U4_LAYOUT
    else:
        raise ValueError(f"unsupported Flash-Next numeric format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


def _hyper_connection(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        tensor_spec(prefix + "block_inject", (4, 10_240), BF16),
        tensor_spec(prefix + "norm", (10_240,), BF16),
        tensor_spec(prefix + "input_mix/down", (320, 10_240), BF16),
        tensor_spec(prefix + "input_mix/up", (10_240, 320), BF16),
    )


def _hyper_connection_mixer(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        tensor_spec(prefix + "norm", (10_240,), BF16),
        tensor_spec(prefix + "input_mix/down", (320, 10_240), BF16),
        tensor_spec(prefix + "input_mix/up", (10_240, 320), BF16),
    )


def _common_mlp(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        tensor_spec(prefix + "router", (512, 2_560), BF16),
        tensor_spec(prefix + "shared_expert/down", (2_560, 640), BF16),
        tensor_spec(prefix + "shared_expert/gate", (640, 2_560), BF16),
        tensor_spec(prefix + "shared_expert/up", (640, 2_560), BF16),
        tensor_spec(prefix + "shared_expert_gate", (1, 2_560), BF16),
    )


def _gdn(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        tensor_spec(prefix + "a_log", (48,), BF16),
        tensor_spec(prefix + "convolution", (4, 10_240), BF16),
        tensor_spec(prefix + "dt_bias", (48,), BF16),
        tensor_spec(prefix + "a_b_projection", (96, 2_560), BF16),
        tensor_spec(prefix + "norm", (128,), BF16),
        tensor_spec(prefix + "query_key_value_z", (16_384, 2_560), FP8),
        tensor_spec(prefix + "output", (2_560, 6_144), FP8),
    )


def _attention(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        tensor_spec(prefix + "indexer/query_key", (640, 2_560), BF16),
        tensor_spec(prefix + "indexer/key_norm", (128,), BF16),
        tensor_spec(prefix + "indexer/query_norm", (128,), BF16),
        tensor_spec(prefix + "key_norm", (256,), BF16),
        tensor_spec(prefix + "query_norm", (256,), BF16),
        # q_proj stores head-major [query, output-gate] row pairs.  K and V
        # follow as two contiguous 512-row regions.
        tensor_spec(prefix + "query_gate_key_value", (13_312, 2_560), FP8),
        tensor_spec(prefix + "output", (2_560, 6_144), FP8),
    )


def _build_text_layer_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for layer in range(48):
        prefix = f"text/layers/{layer}/"
        specs.extend(_hyper_connection(prefix + "attention/hyper_connection/"))
        specs.extend(_common_mlp(prefix + "mlp/"))
        specs.extend(_hyper_connection(prefix + "mlp/hyper_connection/"))
        specs.extend(
            (
                tensor_spec(prefix + "mlp/experts/gate_up", (512, 1_280, 2_560), NVFP4),
                tensor_spec(prefix + "mlp/experts/down", (512, 2_560, 640), NVFP4),
            )
        )
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(_attention(prefix + "attention/"))
        else:
            specs.extend(_gdn(prefix + "gdn/"))
    return tuple(specs)


def _build_ple_specs() -> tuple[TensorSpec, ...]:
    prefix = "text/layers/1/ple/"
    specs: list[TensorSpec] = [
        tensor_spec(prefix + "convolution", (4, 10_240), BF16),
        tensor_spec(prefix + "key_projection", (10_240, 2_560), BF16),
        tensor_spec(prefix + "conv_norm", (10_240,), BF16),
        tensor_spec(prefix + "key_norm", (10_240,), BF16),
        tensor_spec(prefix + "query_norm", (10_240,), BF16),
        tensor_spec(prefix + "value_projection", (2_560, 2_560), BF16),
        tensor_spec(prefix + "embedding/layer_multipliers", (3,), I64),
        tensor_spec(prefix + "embedding/ngram_head_offsets", (16,), I64),
        tensor_spec(prefix + "embedding/ngram_head_vocab_sizes", (16,), I64),
    ]
    specs.extend(
        tensor_spec(prefix + f"embedding/shards/{shard}", (2_500_012, 160), PLE_U4)
        for shard in range(128)
    )
    return tuple(specs)


def _build_global_specs() -> tuple[TensorSpec, ...]:
    return (
        tensor_spec("text/token_embedding", (248_320, 2_560), BF16),
        tensor_spec("text/output_head", (248_320, 2_560), BF16),
        tensor_spec("text/hyper_connection/norm", (10_240,), BF16),
        tensor_spec("text/hyper_connection/input_mix/down", (320, 10_240), BF16),
        tensor_spec("text/hyper_connection/input_mix/up", (10_240, 320), BF16),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("mtp/embedding_projection", (2_560, 2_560), BF16),
        tensor_spec("mtp/hidden_projection", (2_560, 2_560), BF16),
    ]
    specs.extend(_hyper_connection_mixer("mtp/hyper_connection/"))
    specs.extend(_hyper_connection("mtp/layer/attention/hyper_connection/"))
    specs.extend(_common_mlp("mtp/layer/mlp/"))
    specs.extend(_hyper_connection("mtp/layer/mlp/hyper_connection/"))
    specs.extend(
        (
            tensor_spec("mtp/layer/mlp/experts/gate_up", (512, 1_280, 2_560), NVFP4),
            tensor_spec("mtp/layer/mlp/experts/down", (512, 2_560, 640), NVFP4),
            tensor_spec("mtp/layer/attention/indexer/query_key", (640, 2_560), BF16),
            tensor_spec("mtp/layer/attention/indexer/key_norm", (128,), BF16),
            tensor_spec("mtp/layer/attention/indexer/query_norm", (128,), BF16),
            tensor_spec("mtp/layer/attention/key_norm", (256,), BF16),
            tensor_spec("mtp/layer/attention/query_norm", (256,), BF16),
            tensor_spec("mtp/layer/attention/query_gate_key_value", (13_312, 2_560), BF16),
            tensor_spec("mtp/layer/attention/output", (2_560, 6_144), BF16),
            tensor_spec("mtp/embedding_norm", (2_560,), BF16),
            tensor_spec("mtp/hidden_norm", (10_240,), BF16),
        )
    )
    return tuple(specs)


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("vision/patch_embedding", (1_152, 1_536), BF16),
        tensor_spec("vision/patch_embedding_bias", (1_152,), BF16),
        tensor_spec("vision/position_embedding", (2_304, 1_152), BF16),
    ]
    for layer in range(27):
        prefix = f"vision/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "attention/qkv", (3_456, 1_152), BF16),
                tensor_spec(prefix + "attention/qkv_bias", (3_456,), BF16),
                tensor_spec(prefix + "attention/output", (1_152, 1_152), BF16),
                tensor_spec(prefix + "attention/output_bias", (1_152,), BF16),
                tensor_spec(prefix + "mlp/fc1", (4_304, 1_152), BF16),
                tensor_spec(prefix + "mlp/fc1_bias", (4_304,), BF16),
                tensor_spec(prefix + "mlp/fc2", (1_152, 4_304), BF16),
                tensor_spec(prefix + "mlp/fc2_bias", (1_152,), BF16),
                tensor_spec(prefix + "norm1/weight", (1_152,), BF16),
                tensor_spec(prefix + "norm1/bias", (1_152,), BF16),
                tensor_spec(prefix + "norm2/weight", (1_152,), BF16),
                tensor_spec(prefix + "norm2/bias", (1_152,), BF16),
            )
        )
    specs.extend(
        (
            tensor_spec("vision/merger/fc1", (4_608, 4_608), BF16),
            tensor_spec("vision/merger/fc1_bias", (4_608,), BF16),
            tensor_spec("vision/merger/fc2", (2_560, 4_608), BF16),
            tensor_spec("vision/merger/fc2_bias", (2_560,), BF16),
            tensor_spec("vision/merger/norm/weight", (1_152,), BF16),
            tensor_spec("vision/merger/norm/bias", (1_152,), BF16),
        )
    )
    return tuple(specs)


TEXT_LAYER_TENSOR_SPECS = _build_text_layer_specs()
PLE_TENSOR_SPECS = _build_ple_specs()
GLOBAL_TENSOR_SPECS = _build_global_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = _build_vision_specs()
TENSOR_SPECS = (
    TEXT_LAYER_TENSOR_SPECS
    + PLE_TENSOR_SPECS
    + GLOBAL_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_NAMES = (BF16, I64, NVFP4, FP8, PLE_U4)
LAYOUT_NAMES = (
    CONTIGUOUS_LAYOUT,
    EXPERT_NVFP4_LAYOUT,
    FP8_LAYOUT,
    PLE_U4_LAYOUT,
)
FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


def inventory_divergences() -> tuple[str, ...]:
    """Every way this inventory departs from the pinned plan, as messages. Never raises.

    This exists because validate_inventory() raising on the FIRST divergence, from an
    import-time call, made the later checks unreachable. MEASURED 2026-09-18 (FN-GATE):

        python3 -c "import tools.convert.qwen3_8_flash_next.inventory"
        ValueError: unexpected Flash-Next numeric allocation: {'BF16': 1235, 'I64': 3,
          'NVFP4': 98, 'FP8_E4M3FN_ROW_F32S': 96, 'U4Z8G16_F16S': 128}

    The import died at the numeric check, so the layout check below it -- the third known
    divergence -- could never run, and a reader could not tell whether it still held. A preflight
    that wants to REPORT the divergences rather than die on the first one calls this; callers that
    want the old fail-fast behaviour call validate_inventory(), which now raises once with ALL of
    them so nothing is masked by ordering.
    """
    problems: list[str] = []
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        problems.append("Flash-Next inventory contains duplicate object names")
    counts = (
        len(TEXT_LAYER_TENSOR_SPECS),
        len(PLE_TENSOR_SPECS),
        len(GLOBAL_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
    )
    if counts != (1_056, 137, 5, 29, 333, 1_560, 1_566):
        problems.append(
            "Flash-Next persistent inventory is incomplete"
            f" (tensor/object counts {counts}, pinned (1056, 137, 5, 29, 333, 1560, 1566))"
        )
    pinned_formats = {
        BF16: 1_237,
        I64: 3,
        NVFP4: 96,
        FP8: 96,
        PLE_U4: 128,
    }
    if FORMAT_COUNTS != pinned_formats:
        problems.append(
            f"unexpected Flash-Next numeric allocation: {FORMAT_COUNTS}"
            f" (pinned {pinned_formats})"
        )
    pinned_layouts = {
        CONTIGUOUS_LAYOUT: 1_240,
        EXPERT_NVFP4_LAYOUT: 96,
        FP8_LAYOUT: 96,
        PLE_U4_LAYOUT: 128,
    }
    if LAYOUT_COUNTS != pinned_layouts:
        problems.append(
            f"unexpected Flash-Next layout allocation: {LAYOUT_COUNTS}"
            f" (pinned {pinned_layouts})"
        )
    return tuple(problems)


def validate_inventory() -> None:
    """Raise once, naming EVERY divergence instead of only the first one."""
    divergences = inventory_divergences()
    if divergences:
        raise ValueError(
            "Flash-Next inventory does not match the pinned plan:\n  - "
            + "\n  - ".join(divergences)
        )


# Import-time validation, still fail-fast BY DEFAULT, with one documented way past it.
#
# Why the escape exists (MEASURED 2026-09-18, FN-GATE): the whole point of inventory_divergences()
# is that a preflight can REPORT the divergences rather than die on the first one -- but with an
# unconditional raise here the module cannot be imported at all, so the reporter was itself
# unreachable. The default is unchanged (an unset variable validates exactly as before, and every
# existing consumer keeps the fail-fast guarantee), and only the escape is new. Nothing in this
# tree sets the variable; the front door that wants the report sets it explicitly:
#
#   NINFER_FLASHNEXT_ALLOW_UNPINNED_INVENTORY=1 python -c \
#     "from tools.convert.qwen3_8_flash_next import inventory as i; print(*i.inventory_divergences(), sep='\n')"
if os.environ.get("NINFER_FLASHNEXT_ALLOW_UNPINNED_INVENTORY") != "1":
    validate_inventory()


__all__ = [
    "BF16",
    "EXPERT_NVFP4_LAYOUT",
    "FORMAT_COUNTS",
    "FP8",
    "FP8_LAYOUT",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "GLOBAL_TENSOR_SPECS",
    "I64",
    "LAYOUT_COUNTS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "OBJECT_SPECS",
    "PLE_TENSOR_SPECS",
    "PLE_U4",
    "PLE_U4_LAYOUT",
    "RESOURCE_SPECS",
    "ResourceSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_LAYER_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
    "tensor_spec",
    "validate_inventory",
]
