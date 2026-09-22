"""Single front door for bringing a local model into a `.ninfer` artifact.

The registered converters are closed, byte-pinned contracts: each accepts
exactly one checkpoint shape and refuses everything else.  They are correct to
do so, but "refused" is not a usable answer when the input is a legitimate
variant of a registered model, so this module is the layer that decides between
three outcomes and always says which one applies:

* ``runnable`` - a registered target accepts the source; the exact converter
  command is emitted (and run, unless ``--plan-only``).
* ``work-item`` - the model is understood, but something the pipeline needs is
  absent or shaped differently.  The work item is named with the file that
  implements it; nothing is silently approximated.
* ``error`` - the input cannot become a runnable artifact at all (no weights,
  truncated download, a non-decoder checkpoint).  The message names the file and
  the remedy.

Frontend resources are resolved through
``tools.convert.qwen3_6.common.frontend_policy``, which keeps the pinned sha256
separate from the semantic requirement and records every deviation.  This module
never re-derives that policy, so the report and the converter cannot disagree.

Nothing here mutates the source checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

# Run-by-path support.  The front door is normally invoked as
# `python3 tools/convert/import_model.py <source>`, and the registered
# targets/frontend policy are imported as `tools.convert...`; without the
# repository root on sys.path that import fails no matter how good the rest
# of the routing is.  Resolving ROOT from __file__ keeps the tool relocatable
# instead of depending on the caller's working directory or PYTHONPATH.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Targets whose contract this module can evaluate.  Listed explicitly rather
#: than discovered, so a stray directory under tools/convert never becomes
#: routable by accident.
REGISTERED_TARGETS: tuple[str, ...] = (
    # FlashNext (Qwen3.8-Flash-Next).  The converter directory is named after the
    # engine target (qwen4_exp, the checkpoint's own model_type); it delegates to the
    # adopted igorls/ninfer implementation kept under tools/convert/qwen3_8_flash_next.
    "qwen4_exp",
    "qwen3_8_27b",
    "qwen3_6_27b",
    "qwen3_6_35b_a3b",
    "muse_glimmer_30b",
    # Gemma-4-31B (model_type gemma4).  The converter
    # tools/convert/gemma4_31b/{inventory,recipe,convert}.py landed with a
    # front-door contract -- validate_config + SOURCE_QUANT_METHODS
    # ("compressed-tensors") + SUPPLIES_FRONTEND_RESOURCES (False) -- and named
    # this very line as its work item (gemma4_31b/convert.py:144-146).  Without
    # it the front door skipped the target's own validator and reported "no
    # target accepts this source" for a checkpoint whose converter validates it
    # field by field; the memory-only version of this row produced
    # "可导入（runnable）" in dl/gemmaspark/after.txt §A3.
    "gemma4_31b",
    # Ornith-1.5-9B-Q4_K_M's tier.  The engine has registered it since p29/p29b
    # (src/targets/registry.cpp:289-291 `{"qwen3_5_9b", ...}`; a real run printed
    # `ninfer: target registry: 5 rows` including `family=qwen3_5_9b model_id=qwen3.5-9b`),
    # but this list was never extended, so the front door skipped the target's own
    # validator and told the user no target accepted the source.
    "qwen3_5_9b",
)

#: Search roots for *resource files only*.  Checkpoint weights are never mixed
#: across roots.
DEFAULT_RESOURCE_ROOTS: tuple[str, ...] = tuple(
    root for root in os.environ.get("NINFER_RESOURCE_ROOTS", "").split(os.pathsep) if root
)

NINFER_MAGIC = b"NINFER\x00\x02"
GGUF_MAGIC = b"GGUF"

#: Decoder families the engine has a target for.  A source outside this set is
#: still described, but it cannot be routed.
DECODER_FAMILIES: tuple[str, ...] = ("qwen3_5", "qwen3_5_moe", "muse_glimmer",
                                     "gemma4", "qwen4_exp", "qwen4_exp_text")

#: Architecture suffixes that identify a non-decoder checkpoint.
NON_DECODER_MARKERS: tuple[str, ...] = (
    "ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering",
    "ForMaskedLM", "ForImageClassification",
)

#: Architectures that carry their own converter, keyed by the GGUF's declared
#: ``general.architecture`` -- the same key ``tools/convert/convert_runner.py`` uses, so
#: the front door and the auto-conversion chain cannot disagree about who converts what.
#:
#: Why the generic chain is wrong advice for these: ``gguf_extract.py`` materialises a
#: bf16 safetensors intermediate, and for the 5.78 GB ornith file that intermediate is
#: ~18 GB with a 12-14 GB peak on a 22 GB box, which is why the conversion never finished
#: (``tools/convert/qwen3_5_9b/convert.py`` docstring; n7land REPORT.md section 5.3,
#: measured).  The family converter reads the GGUF directly and streams row blocks.
ARCH_OWN_CONVERTER: dict[str, str] = {
    "qwen35": "tools/convert/qwen3_5_9b/convert.py",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def text_config_of(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the text tower config, tolerating a nested multimodal shell."""

    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


# --------------------------------------------------------------------------- #
# sniffing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Intake:
    path: Path
    kind: str
    detail: str


def sniff(path: Path) -> Intake:
    """Decide what the input is by looking at it, never by trusting a caller."""

    if not path.exists():
        return Intake(path, "missing", "路径不存在")
    if path.is_file():
        with path.open("rb") as handle:
            head = handle.read(8)
        if head == NINFER_MAGIC:
            return Intake(path, "ninfer", "已是 .ninfer artifact")
        if head[:4] == GGUF_MAGIC:
            return Intake(path, "gguf", "GGUF 容器")
        return Intake(path, "unknown-file", f"既非 .ninfer 也非 GGUF（前 8 字节 {head!r}）")

    entries = sorted(child.name for child in path.iterdir())
    if any(name.endswith(".ninfer") for name in entries):
        return Intake(path, "ninfer", "目录内含 .ninfer artifact")
    if any(name.endswith(".gguf") for name in entries):
        return Intake(path, "gguf", "目录内含 .gguf")
    if "config.json" not in entries:
        return Intake(path, "empty-dir" if not entries else "unknown-dir",
                      "无 config.json" + ("（空目录）" if not entries else ""))
    return Intake(path, "hf", "HF 目录")


def _gguf_entry(source: Path) -> Path | None:
    """The ``.gguf`` file a directory source stands for, or ``None`` if ambiguous.

    :145-146 accepts a directory whose listing contains ``*.gguf`` (detail
    "目录内含 .gguf 分片") and everything after it treated that directory as the
    container itself, so ``report_gguf`` answered
    ``IsADirectoryError: [Errno 21] Is a directory`` and then advised re-downloading
    -- an I/O-shaped diagnosis for a layout question.  Measured for a directory
    holding exactly one .gguf and for one holding several
    (``dl/importer2/step2/probeB_C.txt``).  Exactly one ``.gguf`` resolves; zero
    cannot reach here (sniff required one); several stay the caller's named
    refusal, because a real GGUF shard set (``*-NNNNN-of-NNNNN.gguf``) is not
    merged anywhere in this tree (``tools/convert/gguf_extract.py`` and
    ``gguf_kquant.py``: 0 hits for the split convention) and no shard set exists
    on this machine to measure that path.
    """

    if source.is_file():
        return source
    found = sorted(child for child in source.glob("*.gguf") if child.is_file())
    return found[0] if len(found) == 1 else None


# --------------------------------------------------------------------------- #
# family
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Family:
    label: str
    is_decoder: bool
    note: str


def classify_family(config: Mapping[str, Any]) -> Family:
    """Name what the checkpoint is, so a non-decoder never gets decoder advice."""

    architectures = config.get("architectures")
    arch = architectures[0] if isinstance(architectures, list) and architectures else ""
    model_type = str(config.get("model_type") or "")
    text = text_config_of(config)

    if "DraftModel" in arch or "draft" in model_type.lower():
        return Family("external-draft", False,
                      "外部草稿模型（不是主模型）；主模型的推测解码靠 dflash2/* 对象，"
                      "而 tools/convert 里没有它的产出器")
    if any(marker in arch for marker in NON_DECODER_MARKERS):
        kind = "重排/分类" if "Classification" in arch else "编码器"
        return Family("encoder", False,
                      f"{kind}模型（{arch}），没有 KV cache / 自回归解码路径，不是文本生成解码器")
    if model_type in DECODER_FAMILIES:
        return Family(model_type, True, f"已注册解码器族（{arch}）")
    if text.get("head_dim") and text.get("num_hidden_layers"):
        return Family(model_type or "unknown-decoder", True,
                      f"解码器，但族 {arch or model_type!r} 未被任何注册 target 接受")
    return Family(model_type or "unknown", False, f"无法判定为解码器（{arch or model_type!r}）")


# --------------------------------------------------------------------------- #
# weights census (index only - never opens a shard payload)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class WeightCensus:
    index_path: Path | None
    shards_present: tuple[str, ...]
    shards_referenced: tuple[str, ...]
    shards_missing: tuple[str, ...]
    shards_empty: tuple[str, ...]
    tensors: int
    by_class: dict[str, int]
    quant_field_names: dict[str, int]
    mtp_keys: tuple[str, ...]
    vision_keys: int
    max_layer: int | None
    single_file: Path | None


def _classify_key(name: str) -> str:
    lowered = name.lower()
    if "vision" in lowered or "visual" in lowered:
        return "vision"
    if "mtp" in lowered or "nextn" in lowered:
        return "mtp"
    if "embed_tokens" in lowered:
        return "embedding"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head"
    for family in ("linear_attn", "self_attn", "mlp", "norm"):
        if family in lowered:
            return family
    return "other"


def _field_of(name: str) -> str:
    lowered = name.lower()
    for field_name in (
        "weight_scale_2", "weight_scale", "weight_global_scale", "input_global_scale",
        "input_scale", "weight_packed", "weight", "bias",
    ):
        if lowered.endswith(field_name):
            return field_name
    return name.rsplit(".", 1)[-1]


def _census_from_names(names: Sequence[str]) -> WeightCensus:
    import re

    by_class: dict[str, int] = {}
    fields: dict[str, int] = {}
    found = [int(match.group(1)) for name in names
             if (match := re.search(r"layers\.(\d+)\.", name))]
    for name in names:
        key = _classify_key(name)
        by_class[key] = by_class.get(key, 0) + 1
        fields[_field_of(name)] = fields.get(_field_of(name), 0) + 1
    return WeightCensus(
        index_path=None, shards_present=(), shards_referenced=(), shards_missing=(),
        shards_empty=(), tensors=len(names), by_class=by_class, quant_field_names=fields,
        mtp_keys=tuple(name for name in names if _classify_key(name) == "mtp"),
        vision_keys=by_class.get("vision", 0), max_layer=max(found) if found else None,
        single_file=None,
    )


def census_weights(source: Path) -> WeightCensus:
    """Read the index (and the directory listing) without touching shard data."""

    shards_present = tuple(sorted(p.name for p in source.glob("*.safetensors")))
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        single = source / "model.safetensors"
        if single.is_file():
            from safetensors import safe_open

            with safe_open(str(single), framework="pt", device="cpu") as handle:
                names = tuple(handle.keys())
            census = _census_from_names(names)
            return WeightCensus(
                index_path=None, shards_present=(single.name,), shards_referenced=(single.name,),
                shards_missing=(), shards_empty=(), tensors=census.tensors,
                by_class=census.by_class, quant_field_names=census.quant_field_names,
                mtp_keys=census.mtp_keys, vision_keys=census.vision_keys,
                max_layer=census.max_layer, single_file=single,
            )
        return WeightCensus(None, shards_present, (), (), (), 0, {}, {}, (), 0, None, None)

    weight_map = dict(load_json(index_path)["weight_map"])
    referenced = tuple(sorted(set(weight_map.values())))
    missing = tuple(name for name in referenced if not (source / name).is_file())
    empty = tuple(name for name in referenced
                  if (source / name).is_file() and (source / name).stat().st_size == 0)
    census = _census_from_names(tuple(weight_map))
    return WeightCensus(
        index_path=index_path, shards_present=shards_present, shards_referenced=referenced,
        shards_missing=missing, shards_empty=empty, tensors=census.tensors,
        by_class=census.by_class, quant_field_names=census.quant_field_names,
        mtp_keys=census.mtp_keys, vision_keys=census.vision_keys,
        max_layer=census.max_layer, single_file=None,
    )


# --------------------------------------------------------------------------- #
# quantisation flavour
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class QuantFlavour:
    method: str
    detail: str


def _group_summary(quant: Mapping[str, Any]) -> str:
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        return ""
    parts: list[str] = []
    for group_name, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        weights = group.get("weights") if isinstance(group.get("weights"), Mapping) else {}
        targets = group.get("targets")
        count = len(targets) if isinstance(targets, (list, tuple, Mapping)) else "?"
        parts.append(
            f"{group_name}[format={group.get('format')}, bits={weights.get('num_bits')}, "
            f"type={weights.get('type')}, strategy={weights.get('strategy')}, "
            f"group={weights.get('group_size')}, scale={weights.get('scale_dtype')}, targets={count}]"
        )
    return "; ".join(parts)


def classify_quant(source: Path, config: Mapping[str, Any]) -> QuantFlavour:
    """Name the quantisation scheme; acceptance is decided by the converters."""

    quant = config.get("quantization_config")
    hf_quant_path = source / "hf_quant_config.json"
    if not isinstance(quant, Mapping):
        if hf_quant_path.is_file():
            hf_quant = load_json(hf_quant_path)
            inner = hf_quant.get("quantization") if isinstance(hf_quant, Mapping) else None
            algo = inner.get("quant_algo") if isinstance(inner, Mapping) else None
            return QuantFlavour("modelopt+sidecar", f"hf_quant_config.json quant_algo={algo}")
        dtype = text_config_of(config).get("dtype") or config.get("torch_dtype")
        return QuantFlavour("native", f"无 quantization_config（dtype={dtype}）")

    method = str(quant.get("quant_method") or quant.get("quant_algo") or "unknown")
    groups = _group_summary(quant)
    if method == "modelopt" or quant.get("quant_algo"):
        ignore = quant.get("ignore") or quant.get("exclude_modules") or []
        detail = (
            f"quant_method=modelopt, quant_algo={quant.get('quant_algo')}"
            + (f", {groups}" if groups else "")
            + f", ignore={len(ignore) if isinstance(ignore, (list, tuple)) else '?'} 项"
            + (f", kv_cache_scheme={quant.get('kv_cache_scheme')}" if quant.get("kv_cache_scheme") else "")
        )
        return QuantFlavour("modelopt", detail)
    if method == "compressed-tensors":
        return QuantFlavour(
            "compressed-tensors",
            f"format={quant.get('format')}, status={quant.get('quantization_status')}"
            + (f", {groups}" if groups else ""),
        )
    return QuantFlavour(method, f"quant_method={method}" + (f", {groups}" if groups else ""))


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TargetVerdict:
    target: str
    entry: str
    status: str            # accepted | config-mismatch | quant-rejected | unavailable
    message: str
    command: list[str] = field(default_factory=list)


def _format_mismatch(exc: ValueError) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if lines and lines[0].startswith("checkpoint config mismatch"):
        return "；".join(lines[1:]) or lines[0]
    return "；".join(lines)


def _find_validator(module: Any, names: Sequence[str]) -> tuple[Any, str] | None:
    """Locate a validator without guessing the target's internal wiring.

    A thin target re-exports nothing: ``qwen3_8_27b/convert.py`` delegates its
    config validation to an imported family module, so the callable must be
    looked for on the module *and* on the modules it holds.
    """

    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate, f"{module.__name__}.{name}"
    for attribute in vars(module).values():
        for name in names:
            candidate = getattr(attribute, name, None)
            if callable(candidate):
                return candidate, f"{getattr(attribute, '__name__', '?')}.{name}"
    return None


def _validator_names(entry: str) -> tuple[str, ...]:
    if entry == "convert":
        return ("validate_config",)
    return ("_validate_quantized_config", "validate_quantized_config", "validate_config")


def target_source_contract(target: str) -> tuple[frozenset, bool]:
    """What a target's own converter consumes from a source.

    Returns ``(quant_methods_it_converts, supplies_its_own_frontend_resources)``.

    Both default to the behaviour that predates this hook -- ``native`` sources
    only, and the artifact's frontend taken from the source's sidecars -- so a
    target that declares nothing is judged exactly as before it existed.  A
    target whose converter reads a *quantised* source directly, or that
    synthesises the artifact's six frontend resources from pinned copies of its
    own, declares it in its ``convert`` module.  That is the same reason
    ``evaluate_targets`` asks the target's validator instead of comparing pins
    here: the front door must not keep a second copy of an answer the target
    already owns, because the two copies then drift and the front door starts
    refusing a model its own target accepts.
    """
    try:
        module = importlib.import_module(f"tools.convert.{target}.convert")
    except Exception:                                 # noqa: BLE001 - reported at ⑥
        return frozenset({"native"}), False
    methods = getattr(module, "SOURCE_QUANT_METHODS", ("native",))
    return frozenset(methods), bool(getattr(module, "SUPPLIES_FRONTEND_RESOURCES", False))


def _arch_owner_geometry(own: str) -> tuple[int, int] | None:
    """The `(block_count, nextn)` the inventory behind an arch-owned converter implements.

    An arch-owned converter is named for the ARCHITECTURE, and one architecture can be
    published at more than one geometry in the same family -- `qwen35` is.  So the
    architecture alone does not decide whether that converter's inventory describes this
    file; the file's own declared `(block_count, nextn)` does (the same pair the target's
    weight-carrying gate opens with).  These numbers are read out of that target's own
    ``inventory`` module rather than restated here -- the rule
    ``_weight_carrying_gate_note`` follows -- so the two statements cannot drift.
    ``None`` leaves today's advice unchanged: it is returned when the inventory cannot be
    imported or does not declare both numbers, because failing to *report* an unknown is
    worse than the unknown.
    """

    try:
        target = (REPO_ROOT / own).parent.name
        module = importlib.import_module(f"tools.convert.{target}.inventory")
    except Exception:                                 # noqa: BLE001
        return None
    layers, draft = getattr(module, "LAYERS", None), getattr(module, "MTP_LAYERS", None)
    if not isinstance(layers, int) or not isinstance(draft, int):
        return None
    return (layers + draft, draft)


def _weight_carrying_gate_note(target: str, config: Mapping[str, Any]) -> str:
    """What the target's only weight-carrying entry will decide, asked of the target.

    A target with no pure-config validator is reported as ``[?]``, which reads as
    "unknown".  It is not unknown: such a target's one entry point reads weights,
    and every such entry opens with a geometry gate whose numbers the target states
    in its own inventory.  Those numbers are read from there rather than restated
    here, and compared against the source's declared layer count, so the front door
    can say which way this source will go instead of apologising.  An unreadable
    inventory yields ``""`` -- the plain message -- never an exception, because
    failing to *report* an unknown is worse than the unknown.
    """

    try:
        inventory = importlib.import_module(f"tools.convert.{target}.inventory")
    except Exception:                                 # noqa: BLE001
        return ""
    layers, draft = getattr(inventory, "LAYERS", None), getattr(inventory, "MTP_LAYERS", None)
    if not isinstance(layers, int) or not isinstance(draft, int):
        return ""
    note = (f"；该入口是带权重的 convert.py（--gguf），其几何门开篇即要求"
            f" (block_count, nextn) == ({layers + draft}, {draft})")
    declared = config.get("num_hidden_layers")
    if isinstance(declared, int):
        note += f"，本源 config 声明 num_hidden_layers={declared}"
        if declared == layers + draft:
            note += "，该门通过（仍需带权重的实测才能定论）"
        else:
            note += "（该门读的是源自己的 block_count），该门必然拒绝"
    return note


def evaluate_targets(
    source: Path, config: Mapping[str, Any], out_path: Path, device: str
) -> list[TargetVerdict]:
    """Ask each registered target's own validator; never duplicate its pins."""

    verdicts: list[TargetVerdict] = []
    for target in REGISTERED_TARGETS:
        try:
            module = importlib.import_module(f"tools.convert.{target}.convert")
        except Exception as exc:                      # noqa: BLE001 - report, never crash
            verdicts.append(TargetVerdict(target, "convert", "unavailable",
                                          f"无法导入：{type(exc).__name__}: {exc}"))
            continue
        found = _find_validator(module, _validator_names("convert"))
        if found is None:
            verdicts.append(TargetVerdict(
                target, "convert", "unavailable",
                "该 target 未提供纯 config 校验入口（需带权重的 preflight）"
                + _weight_carrying_gate_note(target, config)))
            continue
        validator, origin = found
        try:
            validator(config)
        except ValueError as exc:
            verdicts.append(TargetVerdict(target, "convert", "config-mismatch",
                                          f"[{origin}] {_format_mismatch(exc)}"))
            continue
        except Exception as exc:                      # noqa: BLE001
            verdicts.append(TargetVerdict(target, "convert", "unavailable",
                                          f"[{origin}] {type(exc).__name__}: {exc}"))
            continue
        verdicts.append(TargetVerdict(
            target, "convert", "accepted", f"config 逐字段匹配（校验入口 {origin}）",
            ["python3", "-m", f"tools.convert.{target}.convert",
             "--model", str(source), "--out", str(out_path), "--device", device],
        ))
    return verdicts


def evaluate_nvfp4_entry(
    source: Path, config: Mapping[str, Any], targets: Sequence[str], out_path: Path, device: str
) -> list[TargetVerdict]:
    """Evaluate the dual-source NVFP4 entry where a target provides one."""

    verdicts: list[TargetVerdict] = []
    for target in targets:
        module_name = f"tools.convert.{target}.convert_nvfp4"
        try:
            module = importlib.import_module(module_name)
        except Exception:                             # noqa: BLE001 - entry is optional
            continue
        found = _find_validator(module, _validator_names("nvfp4"))
        if found is None:
            verdicts.append(TargetVerdict(target, "convert_nvfp4", "unavailable",
                                          "该 target 未提供纯量化 config 校验入口"))
            continue
        validator, origin = found
        try:
            validator(config)
        except ValueError as exc:
            verdicts.append(TargetVerdict(target, "convert_nvfp4", "quant-rejected",
                                          f"[{origin}] {_format_mismatch(exc)}"))
            continue
        except Exception as exc:                      # noqa: BLE001
            verdicts.append(TargetVerdict(target, "convert_nvfp4", "unavailable",
                                          f"[{origin}] {type(exc).__name__}: {exc}"))
            continue
        basename = getattr(module, "OUTPUT_BASENAME", out_path.name)
        verdicts.append(TargetVerdict(
            target, "convert_nvfp4", "accepted",
            f"量化 config 通过校验（输出 basename 固定为 {basename}；"
            f"--model 必须是未量化的官方源）；入口 {origin}",
            ["python3", "-m", module_name, "--model", "<未量化的官方源>",
             "--quantized-model", str(source),
             "--out", str(out_path.parent / basename), "--device", device],
        ))
    return verdicts


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _rule(title: str) -> None:
    print(f"\n=== {title} ===")


def report_model(config: Mapping[str, Any]) -> None:
    text = text_config_of(config)
    layers = text.get("layer_types")
    histogram: dict[str, int] = {}
    if isinstance(layers, list):
        for entry in layers:
            histogram[str(entry)] = histogram.get(str(entry), 0) + 1
    rope_params = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), Mapping) else {}
    print(f"  architectures    : {config.get('architectures')}")
    print(f"  model_type       : {config.get('model_type')}")
    print(f"  hidden/layers    : {text.get('hidden_size')} / {text.get('num_hidden_layers')}")
    print(f"  heads q/kv/hd    : {text.get('num_attention_heads')} / "
          f"{text.get('num_key_value_heads')} / {text.get('head_dim')}")
    print(f"  vocab            : {text.get('vocab_size')}")
    print(f"  intermediate     : {text.get('intermediate_size')}")
    print(f"  layer_types      : {histogram or '未声明'}")
    print(f"  mtp_num_layers   : {text.get('mtp_num_hidden_layers')}"
          f"  mtp_use_dedicated_embeddings={text.get('mtp_use_dedicated_embeddings')}")
    print(f"  rope             : theta={text.get('rope_theta') or rope_params.get('rope_theta')}"
          f"  mrope={bool(rope_params.get('mrope_section'))}")
    print(f"  max_position     : {text.get('max_position_embeddings')}")
    print(f"  vision_config    : {'有' if isinstance(config.get('vision_config'), Mapping) else '无'}")


def _mismatch_keys(message: str) -> list[str]:
    """Extract the config keys a validator rejected, from its own wording."""

    body = message.split("] ", 1)[-1]
    keys: list[str] = []
    for clause in body.split("；"):
        clause = clause.strip()
        if ": expected" in clause:
            keys.append(clause.split(":", 1)[0].strip())
    return keys


def _mtp_requirement_note() -> list[str]:
    """Which registered targets mandate mtp/* objects, and how many -- asked, not restated.

    The count is what makes the second blocker a measurement: it is produced by
    calling each target's own ``plan_tensors()``, so a target that stops requiring
    MTP stops being named here without this file changing.
    """

    notes: list[str] = []
    for target in REGISTERED_TARGETS:
        try:
            inventory = importlib.import_module(f"tools.convert.{target}.inventory")
            plan = inventory.plan_tensors()
        except Exception:                             # noqa: BLE001
            continue
        mandatory = [spec for spec in plan if spec.name.startswith("mtp/")]
        if not mandatory:
            continue
        source_names = sorted({source[0] for spec in mandatory for source in spec.sources})
        prefixes = sorted({name.split(".")[0] + "." + name.split(".")[1] + "."
                           for name in source_names if name.count(".") >= 2})
        notes.append(
            f"target {target}：plan_tensors() 共 {len(plan)} 个对象，其中 mtp/* = "
            f"{len(mandatory)} 个（LAYERS={getattr(inventory, 'LAYERS', '?')}, "
            f"MTP_LAYERS={getattr(inventory, 'MTP_LAYERS', '?')}）；"
            f"这 {len(mandatory)} 个要 {len(source_names)} 个源张量，"
            + (f"全部落在同一个层前缀 {prefixes[0]}（=LAYERS，写死）下"
               if len(prefixes) == 1 else f"层前缀 {prefixes}")
            + f"；例：{mandatory[0].name} <- "
              f"{[source[0] for source in mandatory[0].sources]}")
    return notes


def _shape_matches_modulo_mtp(
verdicts: Sequence[TargetVerdict]) -> bool:
    """True when some registered target rejects this source only on MTP keys.

    A source without an MTP layer fails the pinned config check on exactly those
    two keys; the geometry itself is registered, so this is a different outcome
    from "no target knows this shape" and must not be reported as one.  Other
    targets legitimately rejecting on architecture does not change that.
    """

    allowed = {"text_config.mtp_num_hidden_layers", "text_config.mtp_use_dedicated_embeddings"}
    for verdict in verdicts:
        if verdict.status != "config-mismatch":
            continue
        keys = set(_mismatch_keys(verdict.message))
        if keys and keys <= allowed:
            return True
    return False


def _hard_error(code: str, lines: Sequence[str]) -> int:
    _rule(f"✗ 结论：无法导入（{code}）")
    for line in lines:
        print(f"  {line}")
    return 2


# --------------------------------------------------------------------------- #
# GGUF: the verdict is read off the tensor table, never asserted from memory
# --------------------------------------------------------------------------- #
def report_gguf(source: Path) -> int:
    """Say what a .gguf contains, then name what is still missing.

    This branch used to be a static refusal: "the GGUF dequantisation path is not
    implemented", listing four parser defects (a 16-byte header read, type-9 arrays
    rejected, type 5 with a 2-byte stride, a ggml type table that called Q4_0 bf16)
    and "there is no K-quant table".  Every one of those was fixed and Q4_K/Q6_K
    were implemented, so the text had become actively false -- while the GUI chain
    on the same machine accepted the same file and said so.  A hardcoded verdict
    that contradicts the reader is worse than no verdict, so none of the numbers
    below are asserted from memory: the reader is run and its answers are printed.
    """

    gguf_kquant = importlib.import_module("tools.convert.gguf_kquant")
    gguf_names = importlib.import_module("tools.convert.gguf_names")

    try:
        kv, tensors, _data_offset = gguf_kquant.read_tensor_table(str(source))
        provenance = gguf_kquant.layout_provenance(str(source))
    except Exception as exc:                          # noqa: BLE001 - report, never crash
        return _hard_error("F5", [
            f"这个 .gguf 读不出来：{type(exc).__name__}: {exc}",
            "补救：确认下载完整（前 24 字节被截断时会直接报出实际字节数；更深处被截断"
            "的典型形状是元数据/张量表处的 struct 解包错误），或换一份完整的 .gguf。",
        ])

    arch = str(kv.get("general.architecture") or "")
    histogram: dict[str, int] = {}
    for _name, _dims, ttype, _off in tensors:
        key = gguf_kquant.type_name(ttype)
        histogram[key] = histogram.get(key, 0) + 1
    decodable = sorted(gguf_kquant.type_name(t) for t in gguf_kquant.DEQUANTIZERS)
    undecodable = sorted(k for k in histogram if k not in decodable)

    _rule("② 张量表（实测，不是模板）")
    print(f"  architecture : {arch or '（未声明）'}")
    print(f"  张量数        : {len(tensors)}")
    print(f"  类型直方图    : {dict(sorted(histogram.items()))}")
    print(f"  本读者能解码  : {decodable}")
    print(f"  张量数据起点  : {provenance['data_offset']}"
          f"（张量表末 {provenance['tensor_table_end']} 向上对齐到 "
          f"{provenance['alignment']}）")

    coverage = None
    try:
        coverage = gguf_names.coverage(kv, tensors, arch)
    except Exception as exc:                          # noqa: BLE001
        print(f"  名称映射      : 无法评估（{type(exc).__name__}: {exc}）")
    if coverage is not None:
        print(f"  名称映射      : {coverage['mapped']} / {coverage['total']} 张量已映射"
              f"（主栈 {coverage['main_layers']} 层 + 草稿块 {coverage['nextn_layers']}）")
        if coverage["unmapped"]:
            print(f"  **无名称规则**: {coverage['unmapped'][:8]}"
                  + (" …" if len(coverage["unmapped"]) > 8 else ""))
        if coverage["collisions"]:
            print(f"  **HF 键冲突**: {coverage['collisions'][:4]}")

    blockers: list[str] = []
    if undecodable:
        blockers.append(f"类型 {undecodable} 在本读者里没有反量化器（能解 {decodable}）")
    if coverage is not None and coverage["unmapped"]:
        blockers.append(f"{len(coverage['unmapped'])} 个张量没有名称规则")
    if coverage is not None and coverage["collisions"]:
        blockers.append(f"{len(coverage['collisions'])} 个 HF 键冲突")

    model_type = ""
    extract = importlib.import_module("tools.convert.gguf_extract")
    if arch in extract.ARCH_TO_MODEL_TYPE:
        model_type = extract.ARCH_TO_MODEL_TYPE[arch][0]

    _rule("③ 结论")
    if blockers:
        return _hard_error("F5", ["字节层就还读不了："] + blockers + [
            "补救：给 tools/convert/gguf_kquant.py 加该类型的反量化器（并登记进 "
            "DEQUANTIZERS），或给 tools/convert/gguf_names.py 的 RULES 加该架构的名称规则。",
        ])

    print("  字节层        : 可读（类型全部可解码、名称全覆盖）")
    print(f"  解包后的族    : {arch!r} -> model_type={model_type or '?'}")
    print(f"  引擎已注册    : targets={list(REGISTERED_TARGETS)}，"
          f"解码器族={list(DECODER_FAMILIES)}")
    if model_type and model_type in DECODER_FAMILIES:
        print("  可导入（runnable）：引擎有该族的 target")
        print("  命令：")
        # Printed only when the script is really on disk: advice that points at a
        # missing file is the "recognised but not doable" shape, not a runnable route.
        own = ARCH_OWN_CONVERTER.get(arch)
        # An arch-owned command is printed only when that converter's inventory describes
        # THIS file.  `coverage` above already derived this file's own (main layers, draft
        # blocks) from its declaration, so the comparison is made instead of assumed: a
        # command whose gate refuses the file by name (exit 3) is the "recognised but not
        # doable" shape the comment below names, not a runnable route.  On a mismatch the
        # route that CAN run the file is printed instead -- the same extract route the
        # `else` below already offers.
        if own is not None:
            declared_geometry = (
                (coverage["main_layers"], coverage["nextn_layers"])
                if coverage is not None else None
            )
            owner_geometry = _arch_owner_geometry(own)
            if (declared_geometry is not None and owner_geometry is not None
                    and declared_geometry != owner_geometry):
                print("  ** this file is not the geometry that converter implements **")
                print(f"     {own} implements (block_count, nextn) == {owner_geometry}")
                print(f"     this file declares (block_count, nextn) == {declared_geometry}"
                      "（源自述，见上面第 3 节）")
                print("     -> 该入口的几何门会按名拒绝本文件（rc=3），上面的命令跑不通。")
                own = None
        if own is not None and (REPO_ROOT / own).is_file():
            print(f"    python3 {own} --gguf {source} "
                  f"--resources <six-resource-dir> --out <out>.ninfer")
            print("      （--resources 需要 tokenizer.json / tokenizer_config.json / "
                  "chat_template.jinja / generation_config.json /")
            print("        preprocessor_config.json / video_preprocessor_config.json "
                  "六个文件；自动编排见 tools/convert/convert_runner.py，"
                  "搜索根用 NINFER_RESOURCE_ROOTS）")
        else:
            print(f"    python3 tools/convert/gguf_extract.py --src {source} --out <dir>")
        return 0
    print("  work-item（字节层已通，引擎侧缺件，不能导入）：")
    print(f"    1) 没有任何已注册 target 接受 model_type={model_type or '?'}"
          f"（GGUF 里的 general.architecture={arch!r}）")
    print("       补救：装配一个 target —— src/targets/<name>/{impl,export}/ + "
          "src/targets/registry.cpp 一行 + tools/convert/<name>/，并把该族加进 "
          "DECODER_FAMILIES。")
    print("       注：**认出格式 != 引擎能跑**；上面每个数字都是跑读者跑出来的。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="模型目录 / .gguf / .ninfer")
    parser.add_argument("--out", type=Path, default=None, help="产物路径（默认 <source>/<model>.ninfer）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true", help="只出结论与命令，不执行")
    parser.add_argument("--resource-root", action="append", default=None,
                        help="前端资源的搜索根（可重复；默认读 $NINFER_RESOURCE_ROOTS）")
    parser.add_argument("--allow-frontend-drift", action="store_true",
                        help="允许无法证明与钉死版本等价的前端资源（偏离会记入报告）")
    parser.add_argument("--json", action="store_true", help="把结论以 JSON 输出")
    args = parser.parse_args(argv)

    source: Path = args.source.resolve()
    out_path = args.out or (source.parent / f"{source.name}.ninfer")
    roots = [Path(p) for p in (args.resource_root or DEFAULT_RESOURCE_ROOTS)]

    # `sniff` runs *before* the resource-root check.  What the input is does not
    # depend on whether the frontend resource roots were configured, and exiting
    # here first meant the default invocation reported nothing at all -- not even
    # "this is a GGUF".  The roots are needed from section 5 on only; their absence
    # is reported there as a named missing piece (F10), which is a usable answer,
    # instead of an `sys.exit` string that reads as a usage error.
    intake = sniff(source)
    _rule("① 输入")
    print(f"  path : {source}")
    print(f"  kind : {intake.kind}  （{intake.detail}）")

    if intake.kind == "ninfer":
        container = importlib.import_module("tools.artifact.container")
        with container.Artifact.open(source) as artifact:
            tensors = [obj for obj in artifact.objects if hasattr(obj, "format")]
            print(f"  identity : {artifact.identity.model_id} / {artifact.identity.weights_id}")
            print(f"  objects  : {len(artifact.objects)}（其中张量 {len(tensors)}）")
            print(f"  formats  : {dict(sorted(Counter(obj.format for obj in tensors).items()))}")
        print("\n  结论：已是 artifact，无需导入。可直接运行：")
        print(f"    build/apps/ninfer {source} --prompt hello --max-new 4 --greedy")
        return 0
    if intake.kind == "empty-dir":
        return _hard_error("F1", [
            f"{source} 为空目录，没有任何文件",
            "补救：重新下载该检查点，或把 source 指向真正存放权重的目录。",
        ])
    if intake.kind in ("missing", "unknown-file", "unknown-dir"):
        return _hard_error("F1", [
            f"{source} 不是模型：{intake.detail}",
            "补救：指向 HF 目录（含 config.json + .safetensors）、.gguf 或 .ninfer。",
        ])
    if intake.kind == "gguf":
        entry = _gguf_entry(source)
        if entry is None:
            candidates = sorted(child.name for child in source.glob("*.gguf")
                                if child.is_file())
            return _hard_error("F1", [
                f"{source} 是目录，里面 {len(candidates)} 个 .gguf：{candidates}",
                "补救：把 source 指向其中的一个 .gguf 文件；本文件不猜哪一个。",
                "注：真正的 GGUF 分片集（*-NNNNN-of-NNNNN.gguf）本树没有合并读取器"
                "（tools/convert/gguf_extract.py 与 gguf_kquant.py 里分片约定 0 命中），"
                "因此也不按分片集读；盘上没有分片集可供实测。",
            ])
        return report_gguf(entry)

    config = load_json(source / "config.json")
    family = classify_family(config)

    _rule("② 模型形状")
    report_model(config)
    print(f"  家族判定         : {family.label}  -> {family.note}")

    if not family.is_decoder:
        return _hard_error("F9", [
            f"该检查点不是本引擎能跑的文本生成解码器：{family.note}",
            "补救：嵌入/重排/分类模型与外部草稿模型不属于导入流水线的适用范围；",
            "      主模型的推测解码请改用 dflash2/* 对象（当前 tools/convert 无产出器）。",
        ])

    _rule("③ 量化形态")
    flavour = classify_quant(source, config)
    print(f"  flavour : {flavour.method}")
    print(f"  detail  : {flavour.detail}")

    _rule("④ 权重清点（只读 index）")
    census = census_weights(source)
    print(f"  index    : {census.index_path.name if census.index_path else '（无，单文件）'}")
    print(f"  分片      : 声明 {len(census.shards_referenced)} 个 / 存在 {len(census.shards_present)} 个")
    if census.shards_missing:
        print(f"  **缺失分片**: {list(census.shards_missing)}")
    if census.shards_empty:
        print(f"  **零长分片**: {list(census.shards_empty)}")
    print(f"  张量数    : {census.tensors}  最大层号: {census.max_layer}")
    print(f"  分类      : {dict(sorted(census.by_class.items()))}")
    print(f"  量化伴随字段: {dict(sorted(census.quant_field_names.items()))}")
    print(f"  MTP 键    : {len(census.mtp_keys)} 个"
          f"{' ' + str(list(census.mtp_keys[:3])) if census.mtp_keys else '（源件没有 MTP 层）'}")
    print(f"  vision 键 : {census.vision_keys} 个")

    if census.tensors == 0 and not census.shards_missing:
        return _hard_error("F1", [
            "config.json 存在但没有任何可达张量（无 index、无 model.safetensors）。",
            "补救：确认下载完整，或把 source 指向真正存放 .safetensors 的目录。",
        ])
    if census.shards_missing or census.shards_empty:
        return _hard_error("F2", [
            f"权重不完整：index 引用了不存在的分片 {list(census.shards_missing)}"
            + (f"，零长分片 {list(census.shards_empty)}" if census.shards_empty else ""),
            "补救：续传/重新下载该检查点；分片名已在上面列出。",
        ])

    _rule("⑤ 前端资源（6 个钉死项）")
    # Two different outcomes, kept apart: no roots configured (F10 -- we cannot
    # check) versus roots configured and a resource rejected.  Neither is a pass.
    counts = Counter()
    profile = None
    frontend_error = None
    frontend_reject = None
    if not roots:
        frontend_error = (
            "没有配置前端资源搜索根（--resource-root 或 $NINFER_RESOURCE_ROOTS），"
            "无法确认该源的 tokenizer / chat template 等 6 个钉死资源齐备")
        print(f"  判定    : **未测量** —— {frontend_error}")
    else:
        frontend = importlib.import_module("tools.convert.qwen3_6.common.frontend_policy")
        profile = frontend.resolve_frontend_profile(source, config, roots=roots)
        counts = Counter(item.status for item in profile.resolutions)
        for item in profile.resolutions:
            mark = {"pinned": "钉死一致", "equivalent": "已登记等价", "consistent": "语义一致",
                    "unproven": "未证实", "missing": "缺失"}[item.status]
            where = f" <- {item.provider}" if item.provider else ""
            print(f"  [{mark}] {item.name}{where}")
            print(f"          证据：{item.evidence}")
        frontend_reject = frontend.acceptability_error(
            profile, allow_unproven=args.allow_frontend_drift)
        print(f"  判定    : {frontend_reject or '可接受'}")

    _rule("⑥ 路由到注册 target")
    verdicts = evaluate_targets(source, config, out_path, args.device)
    accepted = [v for v in verdicts if v.status == "accepted"]
    quant_verdicts = evaluate_nvfp4_entry(
        source, config, [v.target for v in accepted] or list(REGISTERED_TARGETS), out_path, args.device,
    )
    for verdict in verdicts:
        mark = {"accepted": "✓", "config-mismatch": "✗", "unavailable": "?"}[verdict.status]
        print(f"  [{mark}] {verdict.target:<20} {verdict.message}")
    if len(accepted) > 1:
        print(f"  注意：{len(accepted)} 个 target 的 config 契约相同（同形状不同权重族），"
              f"需用 --target 指明；此处默认取 {accepted[0].target}")
    for verdict in quant_verdicts:
        mark = {"accepted": "✓", "quant-rejected": "✗", "unavailable": "?"}[verdict.status]
        print(f"  [{mark}] {verdict.target:<20} nvfp4 入口：{verdict.message}")

    _rule("⑦ 推测解码计划")
    if census.mtp_keys:
        print(f"  MTP 层权重：源件有 {len(census.mtp_keys)} 个 mtp/* 键 -> MTP 可用（draft_tokens 默认 3）")
    else:
        print("  MTP 层权重：**源件没有** -> 只能关闭推测解码，或用外部草稿模型（DFlash2）")
        for line in _mtp_requirement_note():
            print("  " + line)
        print("  ⇒ 本源 MTP 键 = 0 个，这 12 个对象在产物里不可能存在；"
              "存在性由 src/artifact/binder.cpp:29 find_unconsumed 前置要求，缺失即在 :33 抛错。")
    print("  短名单头 text/draft_head：可由仓库排名字典生成，与 MTP 权重无关")

    _rule("⑧ 结论")
    source_methods, supplies_frontend = (
        target_source_contract(accepted[0].target) if accepted
        else (frozenset({"native"}), False))
    frontend_block = frontend_error or frontend_reject
    if supplies_frontend and frontend_block is not None:
        # The source-side six-resource comparison is not evidence about *this*
        # artifact: this target's converter synthesises those resources from
        # pinned copies in its own directory and never reads the source's
        # sidecars for them.  The measurement stays printed in ⑤, and the
        # artifact is still gated on loading and emitting a token below -- so
        # this redirects the check, it does not drop one.
        print(f"  注意：target {accepted[0].target} 自带 artifact 前端资源，"
              f"源侧 sidecar 的钉死比对不作为产物判据（⑤ 的测量保留供对照）。")
        frontend_block = None
    if accepted and flavour.method in source_methods and frontend_block is None:
        chosen = accepted[0]

        def _runnable_json(**extra) -> None:
            """The machine-readable verdict for THIS path, twin of the work-item block.

            The work-item branch used to be the only one that could emit JSON, and its own
            comment said so; a source that can be imported returns above it, so
            ``import_model.py --json`` printed no JSON at all on exactly the sources this
            file calls runnable, and the caller had to read the exit code instead.  The
            keys are the work item's keys plus the ones only this path has: which target,
            which command, whether it ran, and what the artifact said when it was loaded.
            """
            if not args.json:
                return
            payload = {
                "source": str(source), "kind": intake.kind, "family": family.label,
                "quant": flavour.method, "quant_detail": flavour.detail,
                "tensors": census.tensors, "shards_missing": list(census.shards_missing),
                "mtp_keys": len(census.mtp_keys), "vision_keys": census.vision_keys,
                "verdict": "runnable",
                "target": chosen.target,
                "command": chosen.command,
                "resource_roots": [str(r) for r in roots],
                "frontend": profile.report()["resources"] if profile else None,
                "frontend_counts": dict(counts),
                "frontend_verdict": frontend_reject or None,
                "targets": [{"target": v.target, "entry": v.entry, "status": v.status,
                             "message": v.message, "command": v.command}
                            for v in list(verdicts) + list(quant_verdicts)],
            }
            payload.update(extra)
            print()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"  可导入（runnable）：target={chosen.target}，入口=convert.py")
        # `profile` is None when no resource root was configured.  That used to be
        # unreachable here because the branch required a measured frontend; a target
        # that supplies its own frontend resources can now reach it, so the coupling
        # is stated rather than assumed.
        for item in (profile.deviations if profile is not None else ()):
            print(f"  注意：前端资源 {item.name} 为 {item.status}（{item.evidence[:80]}）")
        print("  命令：")
        # `--resource-root` is THIS file's flag; the converter named in the command reads
        # its six frontend resources from its own `--resources` or from
        # $NINFER_RESOURCE_ROOTS.  So the roots this front door just measured against are
        # put INTO the printed command, and the same value is handed to the child process
        # below.  Before this, that line was the one part of this report that could not be
        # run as printed on a machine whose environment did not carry the roots already.
        resource_env = None
        if roots:
            resource_env = dict(os.environ)
            resource_env["NINFER_RESOURCE_ROOTS"] = os.pathsep.join(str(r) for r in roots)
            print("    NINFER_RESOURCE_ROOTS=%s %s"
                  % (shlex.quote(resource_env["NINFER_RESOURCE_ROOTS"]),
                     " ".join(chosen.command)))
            print("    前置：NINFER_RESOURCE_ROOTS 就是 target 的 convert.py 读资源的通道；"
                  "上面的根与 ⑤ 用来测量的根是同一份，子进程按同一份环境启动。")
        else:
            print("    " + " ".join(chosen.command))
            print("    前置：本文件没有配置前端资源搜索根（⑤ 的判定为未测量），这条命令里"
                  "因此不带根；该 target 自带 artifact 前端资源，不读源侧 sidecar。若换成"
                  "读源侧 sidecar 的 target，命令须另加 --resource-root <dir>，或先导出"
                  " NINFER_RESOURCE_ROOTS=<dir>，否则子进程看不到那六个文件。")
        if not args.plan_only:
            print("  正在执行……")
            rc = subprocess.call(chosen.command, cwd=str(REPO_ROOT),
                                   env=resource_env)
            if rc != 0:
                print(f"  转换器 rc={rc}，没有产物。")
                _runnable_json(executed=True, converter_rc=rc, verified=False,
                               artifact_error="converter exited %d; no artifact" % rc)
                return rc
            # rc == 0 says the converter ran; "runnable" is a claim about the
            # artifact, so it is decided by loading it.  The gate is imported from
            # convert_runner rather than re-implemented here, so the CLI chain and
            # the GUI chain cannot reach different verdicts about the same file.
            from tools.convert.convert_runner import ArtifactUnverified, verify_artifact
            print(f"  加载验证 {out_path} ……")
            try:
                artifact_report = verify_artifact(str(out_path))
                print(f"  验证通过：{artifact_report}")
            except ArtifactUnverified as exc:
                print(f"  **转换完成但产物跑不了 —— 不算成功**：{exc}")
                _runnable_json(executed=True, converter_rc=0, verified=False,
                               artifact_error=str(exc))
                return 4
            _runnable_json(executed=True, converter_rc=0, verified=True,
                           artifact=str(artifact_report))
        else:
            _runnable_json(executed=False, verified=None)
        return 0

    print("  work-item（已理解，但缺件/形态不同，需先落地对应实现）：")
    if frontend_error:
        print(f"    0) 前端资源根未配置（硬阻断，本源的 tokenizer 侧车未测量）："
              f"{frontend_error}")
        print("       补救：加 --resource-root <dir>（可重复）或设 "
              "NINFER_RESOURCE_ROOTS=<dir>[,...]。")
    if flavour.method == "modelopt":
        print("    1) ModelOpt NVFP4 单源适配器：源是 modelopt/NVFP4（全部 401 个 Linear），")
        print("       注册布局要求 145 个对象为 FP8、112 个为 NVFP4；")
        print("       实测源布局 = weight U8[N,K/2] + weight_scale F8_E4M3[N,K/16]")
        print("                   + weight_scale_2 F32 标量（乘数，运行时当除数，须取倒数）")
        print("                   + input_scale F32 标量（同上）")
        print("       位置：tools/convert/dequant/modelopt.py + tools/convert/qwen3_8_27b/convert_modelopt.py")
    elif flavour.method == "compressed-tensors" and quant_verdicts and \
            all(v.status == "quant-rejected" for v in quant_verdicts):
        print("    1) compressed-tensors 单 group 变体不被双源入口接受（它要求 group_0 fp8 + group_1 nvfp4）")
    if not census.mtp_keys:
        print("    2) MTP：**按名拒绝（已测量，不是待办）** —— 源提供 0 个 mtp/* 键，"
              "而某个注册 target 无条件要求 12 个。")
        for line in _mtp_requirement_note():
            print("       " + line)
        print("       无条件性：C++ 侧 12 个 bind_mtp( 调用点全部无守卫；"
              "features.mtp() 只决定 TensorPlacement::Device 还是 ValidateOnly，"
              "存在性由 src/artifact/binder.cpp:29 find_unconsumed 前置，"
              "缺失即在 :33 抛 'required artifact object is missing'。")
        print("       而 features.mtp() 来自命令行（EngineOptions.speculative），**不读产物**。")
        print("       ⇒ **拒绝**「MTP 可选化」：本源没有 MTP 是产物的属性；"
              "把它改成可选，只会把同一次失败从加载期推迟到 --spec mtp 期，"
              "并把「声明与产物不一致」藏起来。这是一处引擎侧改动，本行不落地。")
    if profile is not None and profile.missing:
        print(f"    3) 前端资源缺件（硬阻断）：{list(profile.missing)}")
    if not accepted and _shape_matches_modulo_mtp(verdicts):
        print("    注：**形状本身是匹配的**（注册 target 只在 MTP 那两个 config 键上不符）——")
        print("        即上列第 2 项是唯一缺口，不需要生成新 target。")
    elif not accepted:
        print("    4) 该形状没有被任何注册 target 接受 -> 需生成新 target（tools/archkit/adapt_all.py）")
        refuted = [v for v in verdicts if v.status == "config-mismatch"]
        if refuted:
            counted = [(v.target, len(_mismatch_keys(v.message))) for v in refuted]
            total = sum(n for _t, n in counted)
            print(f"       config 驳回 {len(refuted)} 个，共 {total} 处 `键: expected` 子句："
                  + "；".join(f"{t}={n}" for t, n in counted))
            untokenised = [t for t, n in counted if n == 0]
            if untokenised:
                print(f"       另有 {len(untokenised)} 个 target 的驳回是一句话而不是键列表"
                      f"（{untokenised}），上面的 0 **不是「没有异议」**，"
                      "是该驳回没有可数的键子句。")
        unknown = [v.target for v in verdicts if v.status == "unavailable"]
        if unknown:
            print(f"       无纯 config 入口（需带权重的 preflight）：{unknown} —— "
                  "该入口是 convert.py --gguf，几何门在读到第一行权重前就把行列定死；"
                  "本行已实跑：rc=3，stderr 811 B，"
                  "'geometry mismatch: the file declares block_count=64 nextn=0, "
                  "the inventory is written for 32 main + 1 draft'。")
        print("       新 target 的交付物（文件清单，照 src/targets/qwen3_5_9b/ 与 "
              "tools/convert/qwen3_5_9b/ 计）：")
        print("         python 侧 3 个：tools/convert/<name>/{inventory.py, convert.py, "
              "check_bindings.py}（9B 参照 9,399 + 21,062 + 10,716 B）")
        print("         C++ 侧 8 个：src/targets/<name>/CMakeLists.txt；"
              "export/ninfer/targets/<name>/package.h；impl/config.h；impl/package.cpp；")
        print("                     impl/variant.h；impl/variant.cpp；"
              "impl/load/bindings.h；impl/load/bindings.cpp（9B 参照合计 103,108 B）")
        print("         既有文件 4 处：src/CMakeLists.txt 加 add_subdirectory(targets/<name>)；"
              "src/targets/registry.h 加 include + using + Loaded/Instance 对 + "
              "std::variant 备选；")
        print("                      src/targets/registry.cpp 加一行 "
              "{family, model_id, target_key, declares_model, entry}；"
              "tools/convert/import_model.py 的 REGISTERED_TARGETS 加一项。")
        print("         另需**一次编译 + 一次真实加载**才能证明它可用（本行禁用构建与引擎）"
              "⇒ 按名拒绝，本行不落地。")

    if args.json:
        print()
        print(json.dumps({
            "source": str(source), "kind": intake.kind, "family": family.label,
            "quant": flavour.method, "quant_detail": flavour.detail,
            "tensors": census.tensors, "shards_missing": list(census.shards_missing),
            "mtp_keys": len(census.mtp_keys), "vision_keys": census.vision_keys,
            # Both paths emit a block now: the runnable one in section 8 builds its
            # own with verdict "runnable" (see _runnable_json there).  The verdict is
            # stated here rather than left to be inferred from the exit code.
            "verdict": "work-item",
            "frontend": profile.report()["resources"] if profile else None,
            "frontend_counts": dict(counts),
            "frontend_verdict": frontend_error or frontend_reject or None,
            "targets": [{"target": v.target, "entry": v.entry, "status": v.status,
                         "message": v.message, "command": v.command}
                        for v in list(verdicts) + list(quant_verdicts)],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
