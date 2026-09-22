# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
"""Closed source-to-artifact recipe for Qwen3.8-Flash-Next."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import re

from . import inventory
from .source import MIXED_INDEX_TENSORS, SourceError, TensorSlice, read_safetensor_directory


DEFAULT_MIXED_DIR = Path(r"E:\NInfer\qwen3_8_flash_next\source\mixed")


@dataclass(frozen=True, slots=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DirectRecipe:
    object_name: str
    sources: tuple[SourceTensor, ...]
    transform: str = "copy"


@dataclass(frozen=True, slots=True)
class Fp8Recipe:
    object_name: str
    matrices: tuple[SourceTensor, ...]


@dataclass(frozen=True, slots=True)
class ExpertBankRecipe:
    object_name: str
    layer: int
    projections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PleRecipe:
    object_name: str
    shard: int


TensorRecipe = DirectRecipe | Fp8Recipe | ExpertBankRecipe | PleRecipe


@dataclass(frozen=True, slots=True)
class MixedSourceValidation:
    index_tensors: int
    consumed_tensors: int
    replaced_ple_tensors: int
    safetensors: int


def _source(name: str, dtype: str, shape: tuple[int, ...]) -> SourceTensor:
    return SourceTensor(name, dtype, shape)


def _direct(
    recipes: list[DirectRecipe],
    object_name: str,
    source_name: str,
    shape: tuple[int, ...],
    *,
    dtype: str = "BF16",
    transform: str = "copy",
) -> None:
    recipes.append(DirectRecipe(object_name, (_source(source_name, dtype, shape),), transform))


def _hyper_connection(
    recipes: list[DirectRecipe], object_prefix: str, source_prefix: str
) -> None:
    _direct(
        recipes,
        object_prefix + "block_inject",
        source_prefix + "block_inject_weight.weight",
        (4, 10_240),
    )
    _direct(recipes, object_prefix + "norm", source_prefix + "hc_norm.weight", (10_240,))
    _direct(
        recipes,
        object_prefix + "input_mix/down",
        source_prefix + "input_mix_weight_down.weight",
        (320, 10_240),
    )
    _direct(
        recipes,
        object_prefix + "input_mix/up",
        source_prefix + "input_mix_weight_up.weight",
        (10_240, 320),
    )


def _hyper_connection_mixer(
    recipes: list[DirectRecipe], object_prefix: str, source_prefix: str
) -> None:
    _direct(recipes, object_prefix + "norm", source_prefix + "hc_norm.weight", (10_240,))
    _direct(
        recipes,
        object_prefix + "input_mix/down",
        source_prefix + "input_mix_weight_down.weight",
        (320, 10_240),
    )
    _direct(
        recipes,
        object_prefix + "input_mix/up",
        source_prefix + "input_mix_weight_up.weight",
        (10_240, 320),
    )


def _common_mlp(recipes: list[DirectRecipe], object_prefix: str, source_prefix: str) -> None:
    _direct(recipes, object_prefix + "router", source_prefix + "gate.weight", (512, 2_560))
    _direct(
        recipes,
        object_prefix + "shared_expert/down",
        source_prefix + "shared_expert.down_proj.weight",
        (2_560, 640),
    )
    _direct(
        recipes,
        object_prefix + "shared_expert/gate",
        source_prefix + "shared_expert.gate_proj.weight",
        (640, 2_560),
    )
    _direct(
        recipes,
        object_prefix + "shared_expert/up",
        source_prefix + "shared_expert.up_proj.weight",
        (640, 2_560),
    )
    _direct(
        recipes,
        object_prefix + "shared_expert_gate",
        source_prefix + "shared_expert_gate.weight",
        (1, 2_560),
    )


def _fp8_matrix(name: str, shape: tuple[int, int]) -> SourceTensor:
    return _source(name + ".weight", "F8_E4M3", shape)


def _build_recipes() -> tuple[
    tuple[DirectRecipe, ...],
    tuple[Fp8Recipe, ...],
    tuple[ExpertBankRecipe, ...],
    tuple[PleRecipe, ...],
]:
    direct: list[DirectRecipe] = []
    fp8: list[Fp8Recipe] = []
    experts: list[ExpertBankRecipe] = []
    ple: list[PleRecipe] = []

    for layer in range(48):
        obj = f"text/layers/{layer}/"
        src = f"model.language_model.layers.{layer}."
        _hyper_connection(direct, obj + "attention/hyper_connection/", src + "attn_hyper_connection.")
        _common_mlp(direct, obj + "mlp/", src + "mlp.")
        _hyper_connection(direct, obj + "mlp/hyper_connection/", src + "mlp_hyper_connection.")
        experts.extend(
            (
                ExpertBankRecipe(obj + "mlp/experts/gate_up", layer, ("gate_proj", "up_proj")),
                ExpertBankRecipe(obj + "mlp/experts/down", layer, ("down_proj",)),
            )
        )
        if layer in inventory.GDN_LAYERS:
            gp = src + "linear_attn."
            _direct(direct, obj + "gdn/a_log", gp + "A_log", (48,))
            _direct(
                direct,
                obj + "gdn/convolution",
                gp + "conv1d.weight",
                (10_240, 1, 4),
                transform="conv-channel-major",
            )
            _direct(direct, obj + "gdn/dt_bias", gp + "dt_bias", (48,))
            direct.append(
                DirectRecipe(
                    obj + "gdn/a_b_projection",
                    (
                        _source(gp + "in_proj_a.weight", "BF16", (48, 2_560)),
                        _source(gp + "in_proj_b.weight", "BF16", (48, 2_560)),
                    ),
                    "concat-rows",
                )
            )
            _direct(direct, obj + "gdn/norm", gp + "norm.weight", (128,))
            fp8.extend(
                (
                    Fp8Recipe(
                        obj + "gdn/query_key_value_z",
                        (
                            _fp8_matrix(gp + "in_proj_qkv", (10_240, 2_560)),
                            _fp8_matrix(gp + "in_proj_z", (6_144, 2_560)),
                        ),
                    ),
                    Fp8Recipe(
                        obj + "gdn/output",
                        (_fp8_matrix(gp + "out_proj", (2_560, 6_144)),),
                    ),
                )
            )
        else:
            ap = src + "self_attn."
            _direct(
                direct,
                obj + "attention/indexer/query_key",
                ap + "indexer.index_qk_proj.weight",
                (640, 2_560),
            )
            _direct(
                direct,
                obj + "attention/indexer/key_norm",
                ap + "indexer.k_layernorm.weight",
                (128,),
            )
            _direct(
                direct,
                obj + "attention/indexer/query_norm",
                ap + "indexer.q_layernorm.weight",
                (128,),
            )
            _direct(direct, obj + "attention/key_norm", ap + "k_norm.weight", (256,))
            _direct(direct, obj + "attention/query_norm", ap + "q_norm.weight", (256,))
            fp8.extend(
                (
                    Fp8Recipe(
                        obj + "attention/query_gate_key_value",
                        (
                            _fp8_matrix(ap + "q_proj", (12_288, 2_560)),
                            _fp8_matrix(ap + "k_proj", (512, 2_560)),
                            _fp8_matrix(ap + "v_proj", (512, 2_560)),
                        ),
                    ),
                    Fp8Recipe(
                        obj + "attention/output",
                        (_fp8_matrix(ap + "o_proj", (2_560, 6_144)),),
                    ),
                )
            )

    pp = "model.language_model.layers.1.ple."
    po = "text/layers/1/ple/"
    _direct(
        direct,
        po + "convolution",
        pp + "conv1d.weight",
        (10_240, 1, 4),
        transform="conv-channel-major",
    )
    _direct(direct, po + "key_projection", pp + "key_proj.weight", (10_240, 2_560))
    _direct(direct, po + "conv_norm", pp + "norm_conv.weight", (10_240,))
    _direct(direct, po + "key_norm", pp + "norm_key.weight", (10_240,))
    _direct(direct, po + "query_norm", pp + "norm_query.weight", (10_240,))
    _direct(direct, po + "value_projection", pp + "value_proj.weight", (2_560, 2_560))
    _direct(
        direct,
        po + "embedding/layer_multipliers",
        pp + "ple_embedding.layer_multipliers",
        (3,),
        dtype="I64",
        transform="copy",
    )
    _direct(
        direct,
        po + "embedding/ngram_head_offsets",
        pp + "ple_embedding.ngram_heads_offsets",
        (16,),
        dtype="I64",
        transform="copy",
    )
    _direct(
        direct,
        po + "embedding/ngram_head_vocab_sizes",
        pp + "ple_embedding.ngram_heads_vocab_sizes",
        (16,),
        dtype="I64",
        transform="copy",
    )
    ple.extend(
        PleRecipe(po + f"embedding/shards/{shard}", shard) for shard in range(128)
    )

    _direct(
        direct,
        "text/token_embedding",
        "model.language_model.embed_tokens.weight",
        (248_320, 2_560),
    )
    _direct(direct, "text/output_head", "lm_head.weight", (248_320, 2_560))
    _hyper_connection_mixer(
        direct,
        "text/hyper_connection/",
        "model.language_model.hyper_connection_mixer.",
    )

    _direct(direct, "mtp/embedding_projection", "mtp.fc_embedding.weight", (2_560, 2_560))
    _direct(direct, "mtp/hidden_projection", "mtp.fc_hidden.weight", (2_560, 2_560))
    _hyper_connection_mixer(direct, "mtp/hyper_connection/", "mtp.hyper_connection_mixer.")
    _hyper_connection(
        direct,
        "mtp/layer/attention/hyper_connection/",
        "mtp.layers.0.attn_hyper_connection.",
    )
    _common_mlp(direct, "mtp/layer/mlp/", "mtp.layers.0.mlp.")
    _hyper_connection(
        direct,
        "mtp/layer/mlp/hyper_connection/",
        "mtp.layers.0.mlp_hyper_connection.",
    )
    _direct(
        direct,
        "mtp/layer/mlp/experts/gate_up",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        (512, 1_280, 2_560),
    )
    _direct(
        direct,
        "mtp/layer/mlp/experts/down",
        "mtp.layers.0.mlp.experts.down_proj",
        (512, 2_560, 640),
    )
    ma = "mtp.layers.0.self_attn."
    _direct(
        direct,
        "mtp/layer/attention/indexer/query_key",
        ma + "indexer.index_qk_proj.weight",
        (640, 2_560),
    )
    _direct(
        direct,
        "mtp/layer/attention/indexer/key_norm",
        ma + "indexer.k_layernorm.weight",
        (128,),
    )
    _direct(
        direct,
        "mtp/layer/attention/indexer/query_norm",
        ma + "indexer.q_layernorm.weight",
        (128,),
    )
    _direct(direct, "mtp/layer/attention/key_norm", ma + "k_norm.weight", (256,))
    _direct(direct, "mtp/layer/attention/query_norm", ma + "q_norm.weight", (256,))
    direct.append(
        DirectRecipe(
            "mtp/layer/attention/query_gate_key_value",
            (
                _source(ma + "q_proj.weight", "BF16", (12_288, 2_560)),
                _source(ma + "k_proj.weight", "BF16", (512, 2_560)),
                _source(ma + "v_proj.weight", "BF16", (512, 2_560)),
            ),
            "concat-rows",
        )
    )
    _direct(direct, "mtp/layer/attention/output", ma + "o_proj.weight", (2_560, 6_144))
    _direct(direct, "mtp/embedding_norm", "mtp.pre_fc_norm_embedding.weight", (2_560,))
    _direct(direct, "mtp/hidden_norm", "mtp.pre_fc_norm_hidden.weight", (10_240,))

    for layer in range(27):
        vo = f"vision/layers/{layer}/"
        vs = f"model.visual.blocks.{layer}."
        for object_tail, source_tail, shape in (
            ("attention/qkv", "attn.qkv.weight", (3_456, 1_152)),
            ("attention/qkv_bias", "attn.qkv.bias", (3_456,)),
            ("attention/output", "attn.proj.weight", (1_152, 1_152)),
            ("attention/output_bias", "attn.proj.bias", (1_152,)),
            ("mlp/fc1", "mlp.linear_fc1.weight", (4_304, 1_152)),
            ("mlp/fc1_bias", "mlp.linear_fc1.bias", (4_304,)),
            ("mlp/fc2", "mlp.linear_fc2.weight", (1_152, 4_304)),
            ("mlp/fc2_bias", "mlp.linear_fc2.bias", (1_152,)),
            ("norm1/weight", "norm1.weight", (1_152,)),
            ("norm1/bias", "norm1.bias", (1_152,)),
            ("norm2/weight", "norm2.weight", (1_152,)),
            ("norm2/bias", "norm2.bias", (1_152,)),
        ):
            _direct(direct, vo + object_tail, vs + source_tail, shape)
    for object_name, source_name, shape, transform in (
        ("vision/patch_embedding", "model.visual.patch_embed.proj.weight", (1_152, 3, 2, 16, 16), "flatten"),
        ("vision/patch_embedding_bias", "model.visual.patch_embed.proj.bias", (1_152,), "copy"),
        ("vision/position_embedding", "model.visual.pos_embed.weight", (2_304, 1_152), "copy"),
        ("vision/merger/fc1", "model.visual.merger.linear_fc1.weight", (4_608, 4_608), "copy"),
        ("vision/merger/fc1_bias", "model.visual.merger.linear_fc1.bias", (4_608,), "copy"),
        ("vision/merger/fc2", "model.visual.merger.linear_fc2.weight", (2_560, 4_608), "copy"),
        ("vision/merger/fc2_bias", "model.visual.merger.linear_fc2.bias", (2_560,), "copy"),
        ("vision/merger/norm/weight", "model.visual.merger.norm.weight", (1_152,), "copy"),
        ("vision/merger/norm/bias", "model.visual.merger.norm.bias", (1_152,), "copy"),
    ):
        _direct(direct, object_name, source_name, shape, transform=transform)

    return tuple(direct), tuple(fp8), tuple(experts), tuple(ple)


DIRECT_RECIPES, FP8_RECIPES, EXPERT_BANK_RECIPES, PLE_RECIPES = _build_recipes()
RECIPES_BY_OBJECT: dict[str, TensorRecipe] = {
    item.object_name: item
    for group in (DIRECT_RECIPES, FP8_RECIPES, EXPERT_BANK_RECIPES, PLE_RECIPES)
    for item in group
}
MIXED_DIRECT_SOURCES = frozenset(part.name for item in DIRECT_RECIPES for part in item.sources)
MIXED_FP8_SOURCES = frozenset(
    name
    for item in FP8_RECIPES
    for matrix in item.matrices
    for name in (matrix.name, matrix.name + "_scale")
)


_EXPERT_RE = re.compile(
    r"model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<field>weight_packed|weight_scale|weight_global_scale|input_global_scale)"
)
_PLE_RE = re.compile(
    r"model\.language_model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\."
    r"shard_(?P<shard>\d+)\.weight"
)


class _ExpertSources(Collection[str]):
    def __len__(self) -> int:
        return 48 * 512 * 3 * 4

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        match = _EXPERT_RE.fullmatch(value)
        return match is not None and int(match["layer"]) < 48 and int(match["expert"]) < 512

    def __iter__(self) -> Iterator[str]:
        for layer in range(48):
            for expert in range(512):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for field in (
                        "weight_packed",
                        "weight_scale",
                        "weight_global_scale",
                        "input_global_scale",
                    ):
                        yield (
                            f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                            f"{projection}.{field}"
                        )


class _MixedSources(Collection[str]):
    def __len__(self) -> int:
        return len(MIXED_DIRECT_SOURCES) + len(MIXED_FP8_SOURCES) + len(MIXED_EXPERT_SOURCES)

    def __contains__(self, value: object) -> bool:
        return value in MIXED_DIRECT_SOURCES or value in MIXED_FP8_SOURCES or value in MIXED_EXPERT_SOURCES

    def __iter__(self) -> Iterator[str]:
        yield from MIXED_DIRECT_SOURCES
        yield from MIXED_FP8_SOURCES
        yield from MIXED_EXPERT_SOURCES


MIXED_EXPERT_SOURCES: Collection[str] = _ExpertSources()
MIXED_SOURCES: Collection[str] = _MixedSources()


def _require_signature(actual: TensorSlice, expected: SourceTensor) -> None:
    if (actual.dtype, actual.shape) != (expected.dtype, expected.shape):
        raise SourceError(
            f"source signature mismatch for {expected.name}: got "
            f"{(actual.dtype, actual.shape)}, expected {(expected.dtype, expected.shape)}"
        )


def validate_mixed_source(mixed_dir: str | Path) -> MixedSourceValidation:
    mixed = Path(mixed_dir)
    index = json.loads((mixed / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != MIXED_INDEX_TENSORS:
        raise SourceError("mixed source index does not satisfy the pinned tensor count")
    headers = {
        path.name: read_safetensor_directory(path) for path in sorted(mixed.glob("*.safetensors"))
    }

    expected_small = MIXED_DIRECT_SOURCES | MIXED_FP8_SOURCES
    seen_small: set[str] = set()
    expert_count = 0
    replaced: set[int] = set()
    for name, shard in weight_map.items():
        if name in expected_small:
            if shard not in headers or name not in headers[shard].tensors:
                raise SourceError(f"recipe source tensor is unavailable: {name}")
            seen_small.add(name)
        elif name in MIXED_EXPERT_SOURCES:
            if shard not in headers or name not in headers[shard].tensors:
                raise SourceError(f"expert source tensor is unavailable: {name}")
            expert_count += 1
        else:
            match = _PLE_RE.fullmatch(name)
            if match is None:
                raise SourceError(f"mixed source is not allocated by the closed recipe: {name}")
            replaced.add(int(match["shard"]))

    missing = expected_small - seen_small
    if missing:
        raise SourceError(f"mixed source is missing recipe tensor: {sorted(missing)[0]}")
    if expert_count != len(MIXED_EXPERT_SOURCES):
        raise SourceError(f"mixed expert tensor count is {expert_count}; expected {len(MIXED_EXPERT_SOURCES)}")
    if replaced != set(range(128)):
        raise SourceError("mixed source does not contain exactly the 128 replaced BF16 PLE tensors")

    for item in DIRECT_RECIPES:
        for part in item.sources:
            shard = weight_map[part.name]
            _require_signature(headers[shard].tensors[part.name], part)
    for item in FP8_RECIPES:
        for matrix in item.matrices:
            shard = weight_map[matrix.name]
            _require_signature(headers[shard].tensors[matrix.name], matrix)
            scale_name = matrix.name + "_scale"
            scale = SourceTensor(scale_name, "F32", (matrix.shape[0], 1))
            scale_shard = weight_map[scale_name]
            _require_signature(headers[scale_shard].tensors[scale_name], scale)

    return MixedSourceValidation(
        index_tensors=len(weight_map),
        consumed_tensors=len(MIXED_SOURCES),
        replaced_ple_tensors=len(replaced),
        safetensors=len(headers),
    )


def validate_recipe() -> None:
    inventory.validate_inventory()
    if (
        len(DIRECT_RECIPES),
        len(FP8_RECIPES),
        len(EXPERT_BANK_RECIPES),
        len(PLE_RECIPES),
        len(RECIPES_BY_OBJECT),
    ) != (1_240, 96, 96, 128, 1_560):
        raise ValueError("Flash-Next tensor recipe is incomplete")
    if set(RECIPES_BY_OBJECT) != {spec.name for spec in inventory.TENSOR_SPECS}:
        raise ValueError("Flash-Next tensor recipe does not match the inventory")
    if (
        len(MIXED_DIRECT_SOURCES),
        len(MIXED_FP8_SOURCES),
        len(MIXED_EXPERT_SOURCES),
        len(MIXED_SOURCES),
    ) != (1_278, 312, 294_912, 296_502):
        raise ValueError("Flash-Next mixed source allocation is not closed")


validate_recipe()


__all__ = [
    "DEFAULT_MIXED_DIR",
    "DIRECT_RECIPES",
    "DirectRecipe",
    "EXPERT_BANK_RECIPES",
    "ExpertBankRecipe",
    "FP8_RECIPES",
    "Fp8Recipe",
    "MIXED_DIRECT_SOURCES",
    "MIXED_EXPERT_SOURCES",
    "MIXED_FP8_SOURCES",
    "MIXED_SOURCES",
    "MixedSourceValidation",
    "PLE_RECIPES",
    "PleRecipe",
    "RECIPES_BY_OBJECT",
    "SourceTensor",
    "TensorRecipe",
    "validate_mixed_source",
    "validate_recipe",
]
