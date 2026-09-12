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
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Targets whose contract this module can evaluate.  Listed explicitly rather
#: than discovered, so a stray directory under tools/convert never becomes
#: routable by accident.
REGISTERED_TARGETS: tuple[str, ...] = (
    "qwen3_8_27b",
    "qwen3_6_27b",
    "qwen3_6_35b_a3b",
    "muse_glimmer_30b",
)

#: Search roots for *resource files only*.  Checkpoint weights are never mixed
#: across roots.
DEFAULT_RESOURCE_ROOTS: tuple[str, ...] = ("/home/user/models",)

NINFER_MAGIC = b"NINFER\x00\x02"
GGUF_MAGIC = b"GGUF"

#: Decoder families the engine has a target for.  A source outside this set is
#: still described, but it cannot be routed.
DECODER_FAMILIES: tuple[str, ...] = ("qwen3_5", "qwen3_5_moe", "muse_glimmer")

#: Architecture suffixes that identify a non-decoder checkpoint.
NON_DECODER_MARKERS: tuple[str, ...] = (
    "ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering",
    "ForMaskedLM", "ForImageClassification",
)


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
        return Intake(path, "gguf", "目录内含 .gguf 分片")
    if "config.json" not in entries:
        return Intake(path, "empty-dir" if not entries else "unknown-dir",
                      "无 config.json" + ("（空目录）" if not entries else ""))
    return Intake(path, "hf", "HF 目录")


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
            verdicts.append(TargetVerdict(target, "convert", "unavailable",
                                          "该 target 未提供纯 config 校验入口（需带权重的 preflight）"))
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


def _shape_matches_modulo_mtp(verdicts: Sequence[TargetVerdict]) -> bool:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="模型目录 / .gguf / .ninfer")
    parser.add_argument("--out", type=Path, default=None, help="产物路径（默认 <source>/<model>.ninfer）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true", help="只出结论与命令，不执行")
    parser.add_argument("--resource-root", action="append", default=None,
                        help="前端资源的搜索根（可重复；默认 /home/user/models）")
    parser.add_argument("--allow-frontend-drift", action="store_true",
                        help="允许无法证明与钉死版本等价的前端资源（偏离会记入报告）")
    parser.add_argument("--json", action="store_true", help="把结论以 JSON 输出")
    args = parser.parse_args(argv)

    source: Path = args.source.resolve()
    out_path = args.out or (source.parent / f"{source.name}.ninfer")
    roots = [Path(p) for p in (args.resource_root or DEFAULT_RESOURCE_ROOTS)]

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
        return _hard_error("F5", [
            "GGUF 反量化路径尚未实现（tools/convert/gguf_extract.py 已知 4 处解析缺陷：",
            "头长度按 16 字节读而非 20、type 9 数组被拒、type 5 步长 2 而非 4、",
            "ggml 类型表把 Q4_0 当 bf16 且缺 BFloat16），且没有 K-quant 反量化表。",
            "补救：先修 gguf_extract.py（复用 tools/archkit/gguf_tensors.py 的正确读取），",
            "再流式反量化进对象写入器（不要把整份 bf16 落盘）。",
        ])

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
    frontend = importlib.import_module("tools.convert.qwen3_6.common.frontend_policy")
    profile = frontend.resolve_frontend_profile(source, config, roots=roots)
    counts = Counter(item.status for item in profile.resolutions)
    for item in profile.resolutions:
        mark = {"pinned": "钉死一致", "consistent": "语义一致", "unproven": "未证实", "missing": "缺失"}[item.status]
        where = f" <- {item.provider}" if item.provider else ""
        print(f"  [{mark}] {item.name}{where}")
        print(f"          证据：{item.evidence}")
    error = frontend.acceptability_error(profile, allow_unproven=args.allow_frontend_drift)
    print(f"  判定    : {error or '可接受'}")

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
        print("  这是当前阻断本源的缺口之一：注册 target 的 inventory 强制要求 12 个 mtp/* 对象。")
    print("  短名单头 text/draft_head：可由仓库排名字典生成，与 MTP 权重无关")

    _rule("⑧ 结论")
    if accepted and flavour.method == "native":
        chosen = accepted[0]
        print(f"  可导入（runnable）：target={chosen.target}，入口=convert.py")
        for item in profile.deviations:
            print(f"  注意：前端资源 {item.name} 为 {item.status}（{item.evidence[:80]}）")
        print("  命令：")
        print("    " + " ".join(chosen.command))
        if not args.plan_only:
            print("  正在执行……")
            return subprocess.call(chosen.command, cwd=str(REPO_ROOT))
        return 0

    print("  work-item（已理解，但缺件/形态不同，需先落地对应实现）：")
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
        print("    2) MTP 可选化：让 12 个 mtp/* 对象在源缺 MTP 时不再强制")
        print("       （Python inventory/recipe + C++ bindings 的对象闭包），并让 backend 决策读 artifact 实际内容")
    if profile.missing:
        print(f"    3) 前端资源缺件（硬阻断）：{list(profile.missing)}")
    if not accepted and _shape_matches_modulo_mtp(verdicts):
        print("    注：**形状本身是匹配的**（注册 target 只在 MTP 那两个 config 键上不符）——")
        print("        即上列第 2 项是唯一缺口，不需要生成新 target。")
    elif not accepted:
        print("    4) 该形状没有被任何注册 target 接受 -> 需生成新 target（tools/archkit/adapt_all.py）")

    if args.json:
        print()
        print(json.dumps({
            "source": str(source), "kind": intake.kind, "family": family.label,
            "quant": flavour.method, "quant_detail": flavour.detail,
            "tensors": census.tensors, "shards_missing": list(census.shards_missing),
            "mtp_keys": len(census.mtp_keys), "vision_keys": census.vision_keys,
            "frontend": profile.report()["resources"], "frontend_counts": dict(counts),
            "frontend_verdict": error,
            "targets": [{"target": v.target, "entry": v.entry, "status": v.status,
                         "message": v.message, "command": v.command}
                        for v in list(verdicts) + list(quant_verdicts)],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
