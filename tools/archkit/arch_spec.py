#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arch_spec.py — 架构规格的 schema 与工具。

ARCHKIT 的"填表入口": 一份 arch-spec.json 描述一个模型家族的全部接入信息。
本模块提供:
  * SPEC_FIELDS 校验 (字段白名单 + 必填)
  * hf_to_spec(config.json 路径) — 从官方 HuggingFace config 推导规格骨架,
    让"加新架构"从抄 config 开始, 而不是手填。

字段语义见 tools/archkit/_ARCHKIT.md; 几何字段全部照抄 HF config 命名。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 必填 / 可选字段 (生成器消费; 未列字段会被警告)
REQUIRED = ["model_id", "geometry"]
OPTIONAL = [
    "family", "hf", "layer_types", "attention", "mlp", "moe", "gdn",
    "ple", "mtp", "indexer", "vision", "weights", "notes",
]

# 需要浮点/整型转换时用的取值帮助 (值直接照抄 config 即可)
SCALAR_FIELDS = {
    "hidden_size": "hidden", "num_hidden_layers": "layers",
    "num_attention_heads": "query_heads", "num_key_value_heads": "kv_heads",
    "head_dim": "head_dim", "vocab_size": "vocab", "max_position_embeddings": "max_ctx",
    "intermediate_size": "intermediate", "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_eps",
}


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError("spec missing required fields: %s" % missing)
    unknown = [k for k in spec if k not in REQUIRED + OPTIONAL]
    if unknown:
        print("warning: unknown spec fields: %s" % unknown)
    return spec


def hf_to_spec(config_path: str, model_id: str, family: str = "qwen3_6",
               out: str | None = None) -> dict:
    """从 HF config.json 推导 arch-spec (text_config 优先, 兼容多模态壳)。"""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config") or cfg
    geom = {}
    for hf_name, spec_name in SCALAR_FIELDS.items():
        if tc.get(hf_name) is not None:
            geom[spec_name] = tc[hf_name]
    spec = {
        "model_id": model_id,
        "family": family,
        "hf": {
            "model_type": tc.get("model_type", ""),
            "architectures": cfg.get("architectures", []),
        },
        "geometry": geom,
        "notes": "derived from %s" % Path(config_path).name,
    }
    # 层型: 若 config 有 layer_types 就原样带上, 否则按 full_attention_interval 推断
    if isinstance(tc.get("layer_types"), list):
        spec["layer_types"] = tc["layer_types"]
    elif tc.get("full_attention_interval"):
        interval = tc["full_attention_interval"]
        n = tc.get("num_hidden_layers", 0)
        spec["layer_types"] = [
            "full" if (i % interval == 0) else "gdn" for i in range(n)
        ]
    # 各类口味字段 (存在且非 None 才带; None = 该架构无此能力)
    for group, keys in {
        "attention": ["attention_bias", "partial_rotary_factor", "rope_parameters"],
        "moe": ["num_experts", "num_experts_per_tok", "moe_intermediate_size",
                "shared_expert_intermediate_size", "router_aux_loss_coef"],
        "gdn": ["linear_conv_kernel_dim", "linear_key_head_dim",
                "linear_num_key_heads", "linear_num_value_heads",
                "linear_value_head_dim", "output_gate_type"],
        "ple": ["ngram_size", "heads_per_ngram", "ngram_vocab_size_base",
                "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids",
                "split_ngram_parts", "make_ngram_vocab_size_divisible_by"],
        "indexer": ["indexer_budget", "indexer_compress_ratio", "indexer_head_dim",
                    "indexer_kv_heads", "indexer_n_heads", "hc_count",
                    "hc_lowrank"],
        "mtp": ["mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"],
    }.items():
        vals = {k: tc[k] for k in keys if k in tc and tc[k] is not None}
        if vals:
            spec[group] = vals
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("spec written: %s" % out)
    return spec


if __name__ == "__main__":
    # python arch_spec.py extract <config.json> <model_id> [out.json]
    if len(sys.argv) >= 4 and sys.argv[1] == "extract":
        hf_to_spec(sys.argv[2], sys.argv[3],
                   out=sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        print(__doc__)
