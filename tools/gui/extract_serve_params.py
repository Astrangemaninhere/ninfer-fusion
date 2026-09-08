# -*- coding: utf-8 -*-
"""extract_serve_params.py - 从引擎 serve_options.cpp 自动抽取全参数注册表.
数据源唯一事实: usage 文本 + parse 链 => JSON 注册表 (GUI 消费; 引擎加旗标自动出现).
Usage: python3 extract_serve_params.py [--out serve_params.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SRC = Path('/home/user/ninfer-fusion/src/serve/serve_options.cpp')


def extract():
    text = SRC.read_text(encoding='utf-8', errors='replace')

    # usage 行: "--flag ..." 或 "[--flag X]" 到行尾
    usage_block = re.search(r'"usage: .*?"\s*\n(.*?)\n\s*"', text, re.S)

    # 直接扫全部字符串里的旗标模式
    flags = {}

    # 1) usage 里的带说明行 (flag + 默认值说明)
    for m in re.finditer(r'"([^"\\]*--[a-z0-9-]+[^"\\]*)"', text):
        line = m.group(1)
        for fm in re.finditer(r'--([a-z0-9-]+)(?:[ =]+([A-Za-z0-9_|N.F]+))?', line):
            name, arg = fm.group(1), fm.group(2)
            if name in flags:
                continue
            flags[name] = {'flag': '--' + name, 'arg': arg or '', 'takes_value': bool(arg)}

    # 2) parse 链: arg == "--xxx" { value(...) } => 值类型
    for m in re.finditer(r'arg == "([a-z0-9-]+)"\s*\{(.*?)\}', text, re.S):
        name, body = m.group(1), m.group(2)
        f = flags.setdefault(name, {'flag': '--' + name, 'arg': '', 'takes_value': True})
        if 'parse_u32' in body or 'parse_nonnegative_int' in body:
            f['type'] = 'int'
        elif 'parse_f32' in body or 'parse_float' in body:
            f['type'] = 'float'
        elif 'parse_bool' in body or body.strip() == '':
            f['type'] = 'bool'
            f['takes_value'] = False
        else:
            f['type'] = 'string'

    # 3) 默认值说明 (usage 里的 "--xxx defaults to N")
    for name, f in flags.items():
        d = re.search(re.escape(f['flag']) + r'[^"]*defaults to ([^."]+)', text)
        if d:
            f['default'] = d.group(1).strip()
            f.setdefault('type', 'string')

    # 分组归类 (vllm-studio 式分区)
    GROUPS = {
        'core': ['model', 'host', 'port', 'api-key', 'model-id', 'max-context', 'kv-capacity',
                 'max-concurrency', 'max-pending-requests', 'pending-timeout-ms',
                 'default-max-tokens', 'device'],
        'kv': ['kv-dtype', 'kv-layer-storage', 'cold-policy', 'cold-keep-tokens',
               'cold-host-bytes', 'cold-disk-path', 'cold-disk-bytes', 'host-kv-mib',
               'host-state-slots', 'device-state-slots'],
        'speculative': ['spec', 'draft-tokens', 'lm-head-draft'],
        'sampling': ['temperature', 'top-p', 'top-k', 'min-p', 'presence-penalty',
                     'frequency-penalty', 'seed', 'greedy'],
        'thinking': ['no-thinking', 'default-thinking-budget', 'preserve-thinking'],
        'media': ['vision', 'media-cache-mib', 'media-live-mib', 'media-preprocess-threads'],
        'perf': ['no-cuda-graph', 'graph-capture-ceiling', 'prefill-chunk',
                 'no-prefix-reuse', 'no-auto-system-shared-prefix'],
        'network': ['cors', 'max-request-mib'],
        'diagnostics': ['log-stats-interval-ms', 'request-log-jsonl', 'raw-output',
                        'print-token-ids', 'reasoning-effort'],
        'advanced': ['context-cost-presets', 'response-store-max-records',
                     'response-store-max-mib', 'max-private-continuations',
                     'max-shared-prefixes', 'max-long-anchors-per-continuation'],
    }
    grouped = {g: [] for g in GROUPS}
    ungrouped = []
    for name, f in sorted(flags.items()):
        placed = False
        for g, names in GROUPS.items():
            if name in names:
                grouped[g].append(f)
                placed = True
                break
        if not placed:
            ungrouped.append(f)
    return {'total': len(flags), 'groups': grouped, 'ungrouped': ungrouped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='serve_params.json')
    a = ap.parse_args()
    reg = extract()
    Path(a.out).write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding='utf-8')
    print('total flags:', reg['total'])
    for g, items in reg['groups'].items():
        print(' %-12s %2d' % (g, len(items)), [i['flag'] for i in items][:6])
    print(' ungrouped', len(reg['ungrouped']), [i['flag'] for i in reg['ungrouped']][:10])


if __name__ == '__main__':
    main()
