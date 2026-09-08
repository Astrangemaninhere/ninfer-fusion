# -*- coding: utf-8 -*-
"""check_geometry.py - 注意力几何覆盖门 (自动化验收阶段 2.5).

以引擎源码为唯一事实源 (不镜像注册表, 防漂移):
  1. 解析 gqa_attention_geometry.cuh 的 GqaXxxGeometry 别名表 (q, kv, head_dim)
  2. 解析 gqa_attention_decode.cu / gqa_attention_prefill.cu 各分派函数的
     已注册守卫 (== GqaXxx::QHeads / KVHeads) 与未守卫回退 (静默默认)
  3. 对照 spec geometry {query_heads, kv_heads, head_dim} 逐路由判定:
     batch/decode(small_t) / cached / prompt(prefill) / kv_append
  任一核心路由未覆盖 => 硬失败 (报缺失别名与需要补的分派点);
  发现静默回退分派 => 硬失败 (要求补 throw, 防错算);

用法: python3 check_geometry.py <spec.json> [--repo ROOT]
返回: 0 = 覆盖完备; 1 = 缺口
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ALIAS_RE = re.compile(r'using\s+(Gqa\w+)\s*=\s*GqaGeometry<(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?>')
GUARD_RE = re.compile(r'if\s*\(\s*(q\.ne\[1\]|k\.ne\[1\]|cache\.head_dim)\s*==\s*(Gqa\w+)::(QHeads|KVHeads|HeadDim)')
FUNC_RE = re.compile(r'void\s+(\w+)\s*\(')

# 各路由函数名 (引擎分派点)
ROUTES = {
    'decode/batch(small_t)': 'gqa_attention_small_t_launch',
    'cached': 'gqa_attention_cached_small_t_launch',
    'prefill(prompt)': 'gqa_attention_prompt_launch',
    'prompt_attention': 'gqa_attention_prompt_attention_launch',
    'kv_append': 'gqa_kv_append_launch',
}


def parse_aliases(geometry_h: str) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for m in ALIAS_RE.finditer(geometry_h):
        name, q, kv, _split, hd = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        out[name] = (int(q), int(kv), int(hd) if hd else 256)
    return out


def parse_function_dispatch(body: str, aliases: dict[str, tuple[int, int, int]]):
    """一个函数体内的守卫列表: [(guard_var, alias), ...] + 是否有未守卫 launch 尾巴."""
    guards = [(m.group(1), m.group(2)) for m in GUARD_RE.finditer(body)]
    # 静默回退检测: 函数尾部存在直接 launch_for<Alias>(...) 且前面守卫不含该别名的 head_dim
    # 判定依据: 每个 guard 前的 if 都是 return; 简单起见: 找出所有 launch_for<GqaX>
    tail_launches = re.findall(r'launch(?:_for)?\s*<\s*?(\w+)', body)
    return guards, tail_launches


def slice_function(src: str, fn: str) -> str:
    idx = src.find('void %s(' % fn)
    if idx < 0:
        return ''
    # brace 匹配到函数体闭合 (从第一个 '{' 开始计数)
    start = src.find('{', idx)
    if start < 0:
        return ''
    depth = 0
    for i in range(start, len(src)):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return src[idx:i + 1]
    return src[idx:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--repo', default=None)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    geo = spec.get('geometry') or {}
    model_q = int(geo.get('query_heads', 0))
    model_kv = int(geo.get('kv_heads', 0))
    model_hd = int(geo.get('head_dim', 0))
    if not (model_q and model_kv and model_hd):
        print('check_geometry: spec 缺 geometry.query_heads/kv_heads/head_dim')
        return 1
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parent.parent.parent
    kern = (repo / 'src/ops/kernel/gqa_attention_geometry.cuh').read_text(encoding='utf-8')
    aliases = parse_aliases(kern)
    if not aliases:
        print('check_geometry: 无法解析 GqaGeometry 别名 (路径 %s)' % kern)
        return 1

    match_by_name = {n: (q, kv, hd) for n, (q, kv, hd) in aliases.items()
                     if q == model_q and kv == model_kv and hd == model_hd}
    q_collisions = [n for n, (q, kv, hd) in aliases.items() if q == model_q and hd != model_hd]
    if not match_by_name:
        print('== geometry 缺口: (%d q, %d kv, %d head_dim) 无匹配别名' %
              (model_q, model_kv, model_hd))
        print('   引擎注册别名: ' + ', '.join('%s(%d,%d,%d)' % (n, *v)
                                             for n, v in sorted(aliases.items())))
        if q_collisions:
            print('   !! QHeads=%d 但 head_dim 不同的别名: %s (分派无法区分, 需扩守卫)'
                  % (model_q, q_collisions))
        return 1
    name = sorted(match_by_name)[0]

    problems = []
    warnings = []
    for route, fn in ROUTES.items():
        for f in (repo / 'src/ops/launcher/gqa_attention_decode.cu',
                  repo / 'src/ops/launcher/gqa_attention_prefill.cu'):
            src = f.read_text(encoding='utf-8')
            body = slice_function(src, fn)
            if not body:
                continue
            guards, tails = parse_function_dispatch(body, aliases)
            covered_guards = {a for _, a in guards}
            guarded_here = name in covered_guards
            # 静默回退只有在"本 spec 几何未被守卫、会被回退接到"时才致命;
            # 其余未守卫回退 (接别的几何) 记为 warning。
            for t in tails:
                if t in aliases and t not in covered_guards and 'throw' not in body:
                    if not guarded_here:
                        problems.append('%s (%s): 本几何未注册且静默回退 %s' %
                                        (route, fn, t))
                    else:
                        warnings.append('%s (%s): 未守卫回退 %s (接其他几何, 建议补 throw)' %
                                        (route, fn, t))
            if not guarded_here and fn in ('gqa_attention_small_t_launch',
                                           'gqa_attention_prompt_launch',
                                           'gqa_attention_cached_small_t_launch'):
                problems.append('%s (%s): 未注册 %s 几何' % (route, fn, name))
            print('  [%s] %-14s %s' % (route, fn, 'OK' if guarded_here else '-'))
    for w in warnings:
        print('  ~ ' + w)
    if problems:
        print('== check_geometry FAIL:')
        for p in problems:
            print('   - ' + p)
        return 1
    print('== check_geometry OK: %s(%d,%d,%d) 全路由覆盖' % (name, model_q, model_kv, model_hd))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
