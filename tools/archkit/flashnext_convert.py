#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flashnext_convert.py — FlashNext (qwen4_exp) safetensors → ninfer 映射器骨架 (P0)。

契约事实源 = tools/archkit/flashnext_bindings.py (74,520 项, --emit 出 JSON)。
真 checkpoint 未到, 本骨架以契约 + spec 几何驱动, 先把"键 → 形状 → 布局 → 变换"
全部钉死, 真权重到位后只接 reader 与格式钩子:

  ① --plan      每个 engine 张量 → 源键 + artifact 形状/格式/布局/变换名;
                未定项显式 pending (不假装成功), 输出计划 JSON。
  ② --checkpoint 真 checkpoint (单文件或 index 目录) 对计划做双向核对:
                源键缺失 / 形状不符 / 计划外源键, 逐项列出。
  ③ --self-test 合成假 checkpoint (全量键名 + 抽样真张量) 跑全量解析 + 变换单测。

布局事实 (引擎源码实证, 非推测):
  - 2-D 线性权重 artifact = (out, in) = HF Linear 原序, 直通不转置
    (src/targets/qwen3_6_35b_a3b/impl/load/bindings.cpp:126 gdn/convolution {4,8192},
     query_key_value_z {12288,2048} 均为 (out,in))。
  - depthwise conv artifact = (kernel, channels), channels = 2*key_dim + value_dim
    (同 target impl/config.h:37 convolution_dim = 2*key_dim + value_dim 实证)。
  - 1-D 范数直通; token_embd = (vocab, hidden); output_head = (vocab, hidden)。
  - PLE 表 (20019200x160 BF16) 永不进 artifact: SSD sidecar
    (tools/convert/ple_sidecar_build.py, 引擎 PleTable 分页 fault 消费)。

已知契约缺口 (真 checkpoint 到位后按实际键表回填, 本文件用 status 标注):
  - gdn.dt_bias 的 alias 合并了 dt_bias 与 A_log 两个语义不同的源张量, 而引擎
    GdnWeights 两槽都有 (35b bindings.cpp:122-125 a_log/dt_bias) → 需拆成两条契约项。
  - gdn.conv_bias: 引擎 GDN 无 conv bias 槽, 源若有此键需确认并入/忽略。
  - MoE 专家打包布局 (e{x}.gate/up/down) 取决于 MoE 内核工作包, 当前逐专家 2-D
    直通占位, 内核定案后换 pack 变换。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import flashnext_bindings as contract  # noqa: E402

SPEC_PATH = HERE / "specs" / "qwen4_exp_spec.json"

BF16 = "BF16"
CONTIGUOUS = "contiguous-le-v1"

READY = "ready"
PENDING_SHAPE = "pending_shape"
PENDING_SEMANTICS = "pending_semantics"
SIDECAR = "sidecar"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


class Geometry:
    """spec 几何 → 映射器常量 (单一事实源)。"""

    def __init__(self, spec: dict | None = None) -> None:
        spec = spec or _spec()
        g = spec["geometry"]
        self.hidden = int(g["hidden"])
        self.layers = int(g["layers"])
        self.vocab = int(g["vocab"])
        self.q_dim = int(g["query_heads"]) * int(g["head_dim"])
        self.kv_dim = int(g["kv_heads"]) * int(g["head_dim"])
        self.head_dim = int(g["head_dim"])
        gdn = spec["gdn"]
        self.gdn_key_dim = int(gdn["linear_num_key_heads"]) * int(gdn["linear_key_head_dim"])
        self.gdn_value_dim = int(gdn["linear_num_value_heads"]) * int(gdn["linear_value_head_dim"])
        self.gdn_v_heads = int(gdn["linear_num_value_heads"])
        self.gdn_v_head_dim = int(gdn["linear_value_head_dim"])
        self.gdn_conv_kernel = int(gdn["linear_conv_kernel_dim"])
        # 引擎实证: convolution_dim = 2*key_dim + value_dim (35b_a3b config.h:37)
        self.gdn_conv_channels = 2 * self.gdn_key_dim + self.gdn_value_dim
        moe = spec["moe"]
        self.moe_experts = int(moe["num_experts"])
        self.moe_inter = int(moe["moe_intermediate_size"])
        idx = spec["indexer"]
        self.idx_heads = int(idx["indexer_n_heads"])
        self.idx_head_dim = int(idx["indexer_head_dim"])
        self.hc_lowrank = int(idx["hc_lowrank"])
        ple = spec["ple"]
        self.ple_conv_kernel = int(ple["ple_conv_kernel_size"])
        self.ple_row_dim = int(ple["ple_embed_dim"])  # 契约文本为 160, 见 _PLE_TABLE 规则


# ---------------------------------------------------------------------------
# 规则表: engine 名 → (artifact 形状, 变换, 状态, 备注)
# 形状一律用 (out, in) / (kernel, channels) / (dim,) 的 artifact 序书写。
_RULES: list[tuple[re.Pattern[str], str, str, str]] = []


def _rule(pattern: str, transform: str, status: str = READY, note: str = "") -> None:
    _RULES.append((re.compile(pattern), transform, status, note))


# 全局
_rule(r"^token_embd$", "embed")
_rule(r"^output_head$", "linear")
_rule(r"^output_norm$", "norm")

# 逐层公共范数
_rule(r"^layer\.\d+\.attn_norm$", "norm")
_rule(r"^layer\.\d+\.ffn_norm$", "norm")

# GDN (36 层)
_rule(r"^layer\.\d+\.gdn\.in_qkv$", "linear")
_rule(r"^layer\.\d+\.gdn\.conv$", "conv")
_rule(r"^layer\.\d+\.gdn\.conv_bias$", "norm", PENDING_SEMANTICS,
      "引擎 GDN 无 conv bias 槽 (35b bindings 无) → 真键表到位后确认并入 convolution 或忽略")
_rule(r"^layer\.\d+\.gdn\.beta$", "norm")
_rule(r"^layer\.\d+\.gdn\.dt_bias$", "norm", PENDING_SEMANTICS,
      "alias 合并 dt_bias/A_log 两个语义不同张量; 引擎两槽都要 → 契约拆项后按实际键变换")
_rule(r"^layer\.\d+\.gdn\.norm$", "norm")
_rule(r"^layer\.\d+\.gdn\.gate$", "linear")
_rule(r"^layer\.\d+\.gdn\.out$", "linear")

# QSA (12 层) + indexer + hc
_rule(r"^layer\.\d+\.qsa\.q$", "linear")
_rule(r"^layer\.\d+\.qsa\.k$", "linear")
_rule(r"^layer\.\d+\.qsa\.v$", "linear")
_rule(r"^layer\.\d+\.qsa\.o$", "linear")
_rule(r"^layer\.\d+\.qsa\.q_norm$", "norm")
_rule(r"^layer\.\d+\.qsa\.k_norm$", "norm")
_rule(r"^layer\.\d+\.qsa\.idx_wq$", "linear")
_rule(r"^layer\.\d+\.qsa\.idx_wk$", "linear")
_rule(r"^layer\.\d+\.qsa\.idx_norm$", "norm")
_rule(r"^layer\.\d+\.qsa\.hc\.\d+\.down$", "linear")
_rule(r"^layer\.\d+\.qsa\.hc\.\d+\.up$", "linear")

# MoE (48 层): router + shared expert + 512 routed experts
_rule(r"^layer\.\d+\.moe\.router$", "linear")
_rule(r"^layer\.\d+\.moe\.sh_gate$", "linear")
_rule(r"^layer\.\d+\.moe\.sh_up$", "linear")
_rule(r"^layer\.\d+\.moe\.sh_down$", "linear")
_rule(r"^layer\.\d+\.moe\.e\d+\.gate$", "linear", PENDING_SEMANTICS,
      "逐专家 2-D 直通占位; MoE 内核定案后换专家 pack 布局")
_rule(r"^layer\.\d+\.moe\.e\d+\.up$", "linear", PENDING_SEMANTICS,
      "逐专家 2-D 直通占位; MoE 内核定案后换专家 pack 布局")
_rule(r"^layer\.\d+\.moe\.e\d+\.down$", "linear", PENDING_SEMANTICS,
      "逐专家 2-D 直通占位; MoE 内核定案后换专家 pack 布局")

# PLE
_rule(r"^ple\.table$", "sidecar", SIDECAR,
      "20M x160 BF16 永不驻 GPU; SSD sidecar (tools/convert/ple_sidecar_build.py)")
_rule(r"^ple\.(key|value|gate_query)$", "linear")
_rule(r"^ple\.norm$", "norm")
_rule(r"^ple\.conv$", "conv")

# MTP
_rule(r"^mtp\.fc$", "linear")
_rule(r"^mtp\.norm$", "norm")
_rule(r"^mtp\.head_norm$", "norm")


def _ple_table_shape() -> tuple[int, int]:
    """PLE 表行数/行宽以契约声明为准 (spec 的 vocab_base 是 20,000,000, 契约表更大)。"""
    for entry in contract.E:
        if entry["engine"] == "ple.table":
            nums = re.findall(r"\d+", entry["shape"])
            if len(nums) == 2:
                return int(nums[0]), int(nums[1])
    raise ValueError("contract has no ple.table entry")


_PLE_TABLE = _ple_table_shape()


def artifact_shape(engine: str, geo: Geometry) -> tuple[int, ...]:
    """engine 名 → artifact 形状 (实证布局; 见文件头)。"""
    if engine == "token_embd" or engine == "output_head":
        return (geo.vocab, geo.hidden)
    if engine == "output_norm" or re.fullmatch(r"layer\.\d+\.(attn|ffn)_norm", engine):
        return (geo.hidden,)
    if engine.endswith(".gdn.in_qkv"):
        return (geo.gdn_key_dim + geo.gdn_value_dim, geo.hidden)
    if engine.endswith(".gdn.conv"):
        return (geo.gdn_conv_kernel, geo.gdn_conv_channels)
    if engine.endswith(".gdn.conv_bias"):
        return (geo.gdn_conv_channels,)
    if engine.endswith(".gdn.beta") or engine.endswith(".gdn.dt_bias"):
        return (geo.gdn_v_heads,)
    if engine.endswith(".gdn.norm"):
        return (geo.gdn_v_head_dim,)
    if engine.endswith(".gdn.gate"):
        return (geo.gdn_value_dim, geo.hidden)
    if engine.endswith(".gdn.out"):
        return (geo.hidden, geo.gdn_value_dim)
    if engine.endswith(".qsa.q"):
        return (geo.q_dim, geo.hidden)
    if engine.endswith(".qsa.k") or engine.endswith(".qsa.v"):
        return (geo.kv_dim, geo.hidden)
    if engine.endswith(".qsa.o"):
        return (geo.hidden, geo.q_dim)
    if engine.endswith(".qsa.q_norm") or engine.endswith(".qsa.k_norm"):
        return (geo.head_dim,)
    if engine.endswith(".qsa.idx_wq"):
        return (geo.idx_heads * geo.idx_head_dim, geo.hidden)
    if engine.endswith(".qsa.idx_wk"):
        return (geo.idx_head_dim, geo.hidden)
    if engine.endswith(".qsa.idx_norm"):
        return (geo.idx_head_dim,)
    if re.search(r"\.qsa\.hc\.\d+\.down$", engine):
        return (geo.hc_lowrank, geo.hidden)
    if re.search(r"\.qsa\.hc\.\d+\.up$", engine):
        return (geo.hidden, geo.hc_lowrank)
    if engine.endswith(".moe.router"):
        return (geo.moe_experts, geo.hidden)
    if engine.endswith(".moe.sh_gate") or engine.endswith(".moe.sh_up"):
        return (geo.moe_inter, geo.hidden)
    if engine.endswith(".moe.sh_down"):
        return (geo.hidden, geo.moe_inter)
    if re.search(r"\.moe\.e\d+\.(gate|up)$", engine):
        return (geo.moe_inter, geo.hidden)
    if re.search(r"\.moe\.e\d+\.down$", engine):
        return (geo.hidden, geo.moe_inter)
    if engine == "ple.table":
        return _PLE_TABLE
    if engine in ("ple.key", "ple.value", "ple.gate_query"):
        return (geo.hidden, geo.hidden)
    if engine == "ple.norm":
        return (geo.hidden,)
    if engine == "ple.conv":
        return (geo.ple_conv_kernel, geo.hidden)
    if engine == "mtp.fc":
        return (geo.hidden, 2 * geo.hidden)
    if engine == "mtp.norm" or engine == "mtp.head_norm":
        return (geo.hidden,)
    raise KeyError(f"no artifact shape rule for {engine!r}")


def source_shape(engine: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """artifact 形状 → 期望的源张量形状 (HF 侧)。"""
    if engine.endswith(".gdn.conv"):
        # HF depthwise Conv1d: (channels, groups=1, kernel)
        return (shape[1], 1, shape[0])
    return shape


# ---------------------------------------------------------------------------
# 变换 (源张量 → artifact 载荷)


def _check(tensor, shape: tuple[int, ...], label: str):
    import torch

    if tuple(tensor.shape) != shape:
        raise ValueError(f"{label}: shape {tuple(tensor.shape)} != expected {shape}")
    return tensor


def _linear(tensor, shape):
    return _check(tensor, shape, "linear")


def _embed(tensor, shape):
    return _check(tensor, shape, "embed")


def _norm(tensor, shape):
    return _check(tensor, shape, "norm")


def _conv(tensor, shape):
    """HF depthwise conv (channels, 1, kernel) → artifact (kernel, channels)。"""
    if tensor.dim() == 3:
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[2])
    if tuple(tensor.shape) == shape:
        return tensor.contiguous()
    if tuple(tensor.t().shape) == shape:
        return tensor.t().contiguous()
    raise ValueError(f"conv: source {tuple(tensor.shape)} cannot map to {shape}")


def _a_log_to_dt_bias(tensor, shape):
    """A_log → dt_bias 语义变换 (仅在契约拆项后启用, 见文件头缺口说明)。"""
    import torch

    return _check(-torch.exp(tensor.float()), shape, "a_log_to_dt_bias")


TRANSFORMS = {
    "linear": _linear,
    "embed": _embed,
    "norm": _norm,
    "conv": _conv,
    "a_log_to_dt_bias": _a_log_to_dt_bias,
    "sidecar": None,
}


# ---------------------------------------------------------------------------
# 计划


@dataclass(frozen=True)
class PlanEntry:
    engine: str
    source: str
    source_aliases: tuple[str, ...]
    source_shape: tuple[int, ...] | None
    artifact_shape: tuple[int, ...]
    format: str
    layout: str
    transform: str
    status: str
    note: str = ""

    def to_json(self) -> dict:
        out = {
            "engine": self.engine,
            "source": self.source,
            "artifact_shape": list(self.artifact_shape),
            "format": self.format,
            "layout": self.layout,
            "transform": self.transform,
            "status": self.status,
        }
        if self.source_shape is not None:
            out["source_shape"] = list(self.source_shape)
        if self.note:
            out["note"] = self.note
        return out


def _rule_for(engine: str) -> tuple[str, str, str]:
    for pattern, transform, status, note in _RULES:
        if pattern.match(engine):
            return transform, status, note
    raise KeyError(f"no layout rule for engine tensor {engine!r}")


def build_plan(geo: Geometry | None = None, format_override: str = BF16) -> list[PlanEntry]:
    geo = geo or Geometry()
    plan: list[PlanEntry] = []
    for entry in contract.E:
        engine = entry["engine"]
        transform, status, note = _rule_for(engine)
        shape = artifact_shape(engine, geo)
        if transform == "sidecar":
            src_shape = None
        else:
            src_shape = source_shape(engine, shape)
        plan.append(PlanEntry(
            engine=engine,
            source=entry["alias"][0],
            source_aliases=tuple(entry["alias"]),
            source_shape=src_shape,
            artifact_shape=shape,
            format=BF16 if transform != "sidecar" else "SIDECAR",
            layout=CONTIGUOUS if transform != "sidecar" else "raw-bytes-v1",
            transform=transform,
            status=status,
            note=note,
        ))
    return plan


def plan_summary(plan: list[PlanEntry]) -> dict:
    by_status: dict[str, int] = {}
    for item in plan:
        by_status[item.status] = by_status.get(item.status, 0) + 1
    return {
        "entries": len(plan),
        "by_status": by_status,
        "artifact_tensors": sum(1 for p in plan if p.status != SIDECAR),
        "sidecars": sum(1 for p in plan if p.status == SIDECAR),
        "pending": sum(1 for p in plan if p.status.startswith("pending")),
    }


# ---------------------------------------------------------------------------
# 源键解析 (契约 alias → 实际 checkpoint 键)


class SourceIndex:
    """源键索引: 精确命中 O(1); 含通配符的 alias 走 fnmatch 扫描 (慢路径)。"""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self._exact = {name: name for name in names}

    def resolve(self, aliases: tuple[str, ...]) -> str | None:
        for pattern in aliases:
            if pattern in self._exact:
                return pattern
        for pattern in aliases:
            if any(ch in pattern for ch in "*?["):
                for name in self.names:
                    if fnmatch.fnmatch(name, pattern):
                        return name
        return None


def verify_plan(plan: list[PlanEntry], names: list[str],
                shapes: dict[str, tuple[int, ...]] | None = None) -> dict:
    """计划 × checkpoint 双向核对: 缺失 / 形状不符 / 计划外源键。"""
    index = SourceIndex(names)
    missing: list[str] = []
    sidecar_missing: list[str] = []
    mismatched: list[dict] = []
    consumed: set[str] = set()
    for item in plan:
        source = index.resolve(item.source_aliases)
        if item.status == SIDECAR:
            # sidecar 不进 artifact, 但源键同样要认领 (否则误报计划外键)
            if source is None:
                sidecar_missing.append(item.engine)
            else:
                consumed.add(source)
            continue
        if source is None:
            missing.append(item.engine)
            continue
        consumed.add(source)
        if shapes is not None and item.source_shape is not None:
            actual = shapes.get(source)
            if actual is not None and tuple(actual) != item.source_shape:
                mismatched.append({
                    "engine": item.engine,
                    "source": source,
                    "expected": list(item.source_shape),
                    "actual": list(actual),
                })
    unmatched = [name for name in names if name not in consumed]
    return {
        "checked": sum(1 for item in plan if item.status != SIDECAR),
        "missing_sources": missing[:40],
        "missing_count": len(missing),
        "sidecar_missing": sidecar_missing,
        "shape_mismatches": mismatched[:40],
        "mismatch_count": len(mismatched),
        "unconsumed_sources": unmatched[:40],
        "unconsumed_count": len(unmatched),
    }


# ---------------------------------------------------------------------------
# checkpoint 读取 (真权重到位后启用; 此处只读名字与形状, 不载入数据)


def read_checkpoint(path: str | Path) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    """单文件 .safetensors 或含 model.safetensors.index.json 的目录 → (names, shapes)。"""
    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_map: dict[str, str] = dict(index["weight_map"])
        shapes: dict[str, tuple[int, ...]] = {}
        by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        for shard, shard_names in by_shard.items():
            with safe_open(str(path / shard), framework="pt", device="cpu") as handle:
                for name in shard_names:
                    shapes[name] = tuple(handle.get_slice(name).get_shape())
        return list(weight_map), shapes
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in names}
    return names, shapes


# ---------------------------------------------------------------------------
# 自测


def _synthetic_names(canonical_only: bool = True) -> list[str]:
    """契约 alias 展开成合成源键表 (真 checkpoint 未到时的覆盖率基线)。

    canonical_only=True 取每条契约项的首别名 (= 该张量的规范源键);
    False 展开全部别名 (HF + llama.cpp 双风格, 用于别名唯一性检查)。
    """
    names: list[str] = []
    seen: set[str] = set()
    for entry in contract.E:
        aliases = entry["alias"][:1] if canonical_only else entry["alias"]
        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                names.append(alias)
    return names


def _alias_duplicates() -> list[str]:
    """同一源键被多条契约项声明 = 解析顺序依赖 (契约卫生问题)。"""
    owner: dict[str, str] = {}
    dupes: list[str] = []
    for entry in contract.E:
        for alias in entry["alias"]:
            if alias in owner and owner[alias] != entry["engine"]:
                dupes.append(f"{alias}: {owner[alias]} / {entry['engine']}")
            owner.setdefault(alias, entry["engine"])
    return dupes


def _sample_engines() -> list[str]:
    geo = Geometry()
    sample = ["token_embd", "output_head", "output_norm",
              "layer.0.attn_norm", "layer.0.ffn_norm"]
    gdn_layer = contract.GDN_LAYERS[0]
    sample += [f"layer.{gdn_layer}.gdn.{s}" for s in
               ("in_qkv", "conv", "conv_bias", "beta", "dt_bias", "norm", "gate", "out")]
    qsa_layer = contract.QSA_LAYERS[0]
    sample += [f"layer.{qsa_layer}.qsa.{s}" for s in
               ("q", "k", "v", "o", "q_norm", "k_norm", "idx_wq", "idx_wk", "idx_norm")]
    sample += [f"layer.{qsa_layer}.qsa.hc.{g}.{p}" for g in range(2) for p in ("down", "up")]
    sample += [f"layer.0.moe.{s}" for s in ("router", "sh_gate", "sh_up", "sh_down")]
    sample += [f"layer.0.moe.e{x}.{p}" for x in range(2) for p in ("gate", "up", "down")]
    sample += ["ple.key", "ple.value", "ple.norm", "ple.gate_query", "ple.conv",
               "mtp.fc", "mtp.norm", "mtp.head_norm"]
    assert len(sample) == len(set(sample))
    return sample


def _materialize(engine: str, geo: Geometry, index: int):
    """合成源张量: 线性/范数 = artifact 形状, conv = HF depthwise (channels, 1, kernel)。"""
    import torch

    shape = artifact_shape(engine, geo)
    src_shape = source_shape(engine, shape)
    torch.manual_seed(index + 1)
    if engine.endswith(".gdn.conv"):
        return torch.arange(float(src_shape[0] * src_shape[2]),
                            dtype=torch.bfloat16).reshape(src_shape)
    return torch.randn(src_shape, dtype=torch.bfloat16)


def self_test(with_safetensors: bool = False) -> int:
    import torch

    geo = Geometry()
    plan = build_plan(geo)
    summary = plan_summary(plan)
    print(f"[self-test] plan: {summary['entries']} entries "
          f"({summary['artifact_tensors']} tensors + {summary['sidecars']} sidecar), "
          f"pending={summary['pending']}")
    assert summary["entries"] == 74520, summary
    assert summary["sidecars"] == 1
    assert all(p.status != PENDING_SHAPE for p in plan), "shape rules incomplete"

    # 1) 全量键名覆盖: 契约首别名展开 → 每条计划都能解析, 无计划外键
    names = _synthetic_names(canonical_only=True)
    report = verify_plan(plan, names)
    assert report["missing_count"] == 0, report["missing_sources"][:5]
    assert report["unconsumed_count"] == 0, report["unconsumed_sources"][:5]
    print(f"[self-test] coverage: {report['checked']} planned tensors resolved, "
          f"0 missing, 0 unconsumed (canonical source names: {len(names)})")

    # 1b) 别名唯一性: 一个源键只能属于一条契约项, 否则解析结果依赖顺序
    dupes = _alias_duplicates()
    assert not dupes, f"alias collisions: {dupes[:5]}"
    print(f"[self-test] alias uniqueness: {len(_synthetic_names(False))} aliases, 0 collisions")

    # 2) 抽样真变换: 形状 + 数值 + conv 布局顺序
    sample = _sample_engines()
    by_engine = {p.engine: p for p in plan}
    checked = 0
    for i, engine in enumerate(sample):
        item = by_engine[engine]
        tensor = _materialize(engine, geo, i)
        if engine.endswith(".gdn.conv"):
            out = _conv(tensor, item.artifact_shape)
            assert tuple(out.shape) == item.artifact_shape, engine
            # artifact (kernel, channels): 第 k 行 == 源第 k 列
            for k in range(tensor.shape[2]):
                assert torch.equal(out[k], tensor[:, 0, k]), engine
            checked += 1
            continue
        out = TRANSFORMS[item.transform](tensor, item.artifact_shape)
        assert tuple(out.shape) == item.artifact_shape, f"{engine}: {tuple(out.shape)}"
        if item.transform in ("linear", "embed", "norm"):
            assert torch.equal(out, tensor), engine
        checked += 1
    print(f"[self-test] transforms: {checked} sampled entries OK (shape + numeric identity)")

    # 3) PLE 表: sidecar 声明, 永不进 artifact
    ple = by_engine["ple.table"]
    rows, width = ple.artifact_shape
    assert ple.status == SIDECAR and ple.transform == "sidecar"
    assert rows == 20019200 and width == 160, ple.artifact_shape
    print(f"[self-test] ple.table sidecar: {rows}x{width} BF16 "
          f"({rows * width * 2 / 2**30:.1f} GiB, SSD only)")

    # 4) 真 safetensors 读写路径 (小样本文件, 验证 reader + 形状核对)
    if with_safetensors:
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            tensors = {}
            for i, engine in enumerate(sample):
                item = by_engine[engine]
                if item.source_shape is None:
                    continue
                tensors[item.source] = _materialize(engine, geo, i).contiguous()
            path = Path(tmp) / "model.safetensors"
            save_file(tensors, str(path))
            names, shapes = read_checkpoint(path)
            report = verify_plan(plan, names, shapes)
            assert report["mismatch_count"] == 0, report["shape_mismatches"][:3]
            assert report["checked"] - report["missing_count"] == len(tensors)
            print(f"[self-test] safetensors path: {len(tensors)} tensors written, "
                  f"{report['checked'] - report['missing_count']} verified, "
                  f"0 shape mismatches (missing {report['missing_count']} = 计划其余项)")

    print("[self-test] PASS")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="输出映射计划")
    ap.add_argument("--out", metavar="JSON", help="计划写盘路径 (默认只打印摘要)")
    ap.add_argument("--checkpoint", metavar="PATH",
                    help="真 checkpoint (单 .safetensors 或 index 目录) → 计划核对")
    ap.add_argument("--audit", metavar="INDEX_JSON", help="契约双向审计 (转 flashnext_bindings)")
    ap.add_argument("--self-test", action="store_true", help="合成 checkpoint 自测")
    ap.add_argument("--with-safetensors", action="store_true",
                    help="自测附加 safetensors 读写路径 (需 safetensors 已装)")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.with_safetensors)

    if args.audit:
        index = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        names = list(index.get("weight_map", index).keys())
        report = contract.audit(names)
        json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
        print()
        return 0 if report["complete"] else 1

    if args.plan or args.checkpoint:
        plan = build_plan()
        summary = plan_summary(plan)
        print(f"plan: {summary['entries']} entries, {summary['pending']} pending, "
              f"{summary['sidecars']} sidecar")
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"spec": str(SPEC_PATH), "summary": summary,
                 "entries": [p.to_json() for p in plan]},
                indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {args.out}")
        if args.checkpoint:
            names, shapes = read_checkpoint(args.checkpoint)
            report = verify_plan(plan, names, shapes)
            json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
            print()
            return 0 if (report["missing_count"] == 0 and report["mismatch_count"] == 0) else 1
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
