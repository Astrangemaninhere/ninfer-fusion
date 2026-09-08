#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_stubs_v2.py — ARCHKIT v2: 从 arch-spec 生成 bindings/package/recipe
stub 骨架 (v1 已生成 config.h + manifest)。stub 含明确 ARCHKIT 标记与 TODO,
供接入首个目标 (qwen4-exp) 时对照 qwen3_6_27b 真实实现回填; 本生成器随后
按回填差异收敛 (见 _ARCHKIT.md 第 6 节)。

用法: python gen_stubs_v2.py <spec.json> --out <dir>
产物 (每文件带生成头注释):
  <ns>/impl/load/bindings.cpp.stub   权重绑定骨架 (hf 键 -> 引擎语义占位)
  <ns>/impl/package.cpp.stub        identity 注册骨架
  tools/convert/<variant>/recipe_<variant>.py.stub  HF 键提取骨架
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_spec  # noqa: E402


def ns_of(model_id: str) -> str:
    return model_id.replace('-', '_')


def gen_bindings_stub(spec: dict) -> str:
    mid = spec["model_id"]
    ns = ns_of(mid)
    g = spec["geometry"]
    return f"""// ARCHKIT v2 stub — {mid} 权重绑定骨架 (对照 src/targets/qwen3_6_27b/impl/load/bindings.cpp 回填)
// TODO(ARCHKIT): 每个 bind_* 调用点需按 qwen3_6 家族真实绑定签名核对。
#include <ninfer/targets/{ns}/package.h>
#include "targets/{ns}/impl/config.h"

namespace ninfer::targets::{ns}::detail {{

// 几何 (已由 gen_target v1 生成, 此处引用): hidden={g.get('hidden')} layers={g.get('layers')}
// TODO(ARCHKIT): 按 spec['hf'] 的键表展开权重映射:
//   spec.weights 未填时先用 HF 键约定 (q/k/v/o/gate/up/down/norm/embed/lm_head)
// 示例骨架 (Qwen 系惯例, 需按 {mid} 官方 safetensors index 核对):
//   bind_tensor(binder, "model.embed_tokens.weight", ...);
//   for (layer in layers): qkv / o / gate_up / down / norms
//   bind_tensor(binder, "lm_head.weight", ...);

}} // namespace ninfer::targets::{ns}::detail
"""


def gen_package_stub(spec: dict) -> str:
    mid = spec["model_id"]
    ns = ns_of(mid)
    return f"""// ARCHKIT v2 stub — {mid} package 注册骨架 (对照 qwen3_6_27b/impl/package.cpp)
// TODO(ARCHKIT): model_id 常量 / resolve_weights / resolved_auto_speculative /
// sampling_defaults / plan_load 各函数体回填。
#include <ninfer/targets/{ns}/package.h>

namespace ninfer::targets::{ns} {{
// TODO(ARCHKIT): static constexpr std::string_view model_id = "{mid}";
// WeightsProfile::resolve_weights 按 (model_id, weights_id) 映射;
// registry 登记 (src/targets/registry.cpp) 使 artifact 身份可解析。
}} // namespace ninfer::targets::{ns}
"""


def gen_recipe_stub(spec: dict) -> str:
    mid = spec["model_id"]
    variant = ns_of(mid)
    hf = spec.get("hf", {})
    return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recipe_{variant}.py — ARCHKIT v2 stub: {mid} 转换 recipe 骨架.

TODO(ARCHKIT): 对照 tools/convert/qwen3_8_27b/recipe_nvfp4.py 回填:
  1. HF 键清单 (hf: {json.dumps(hf, ensure_ascii=False)})
  2. 量化管线选择 (groupwise-int / nvfp4 / fp8) —— 当前引擎量化器在
     tools/convert/common/quantize.py
  3. 产物 .ninfer 对象名必须与 bindings stub 一致
本文件是骨架: 结构可跑, 张量映射未实现, 勿直接用于转换。
"""
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    spec = arch_spec.load_spec(args.spec)
    ns = ns_of(spec['model_id'])
    out = Path(args.out)
    d1 = out / ns / 'impl' / 'load'
    d2 = out / ns / 'impl'
    d3 = out / 'tools' / 'convert' / ns
    for d in (d1, d2, d3):
        d.mkdir(parents=True, exist_ok=True)
    (d2 / 'package.cpp.stub').write_text(gen_package_stub(spec), encoding='utf-8')
    (d1 / 'bindings.cpp.stub').write_text(gen_bindings_stub(spec), encoding='utf-8')
    (d3 / ('recipe_%s.py.stub' % ns)).write_text(gen_recipe_stub(spec), encoding='utf-8')
    print('stubs -> %s (%s)' % (out, spec['model_id']))
    for f in sorted(out.rglob('*.stub')):
        print('  %s (%d B)' % (f, f.stat().st_size))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
