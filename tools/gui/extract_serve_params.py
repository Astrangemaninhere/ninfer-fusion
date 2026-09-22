# -*- coding: utf-8 -*-
"""extract_serve_params.py - 从引擎 serve_options.cpp 自动抽取全参数注册表.

数据源唯一事实: usage 文本 + parse 链 => JSON 注册表 (GUI 消费; 引擎加旗标自动出现).

Usage:
    python3 extract_serve_params.py [--out serve_params.json] [--src PATH]
    NINFER_SERVE_SRC=/path/to/serve_options.cpp python3 extract_serve_params.py

路径优先级: --src > $NINFER_SERVE_SRC > DEFAULT_SRC(构建机上的源码树).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# 引擎源码默认位置(构建机 WSL 源码树); 可用 --src / $NINFER_SERVE_SRC 覆盖.
DEFAULT_SRC = '/home/user/ninfer-fusion/src/serve/serve_options.cpp'
SRC = Path(DEFAULT_SRC)  # 兼容旧引用

# 分组归类 (vllm-studio 式分区). 只放引擎真实存在的旗标.
GROUPS = {
    'core': ['model', 'host', 'port', 'api-key', 'model-id', 'max-context', 'kv-capacity',
             'max-concurrency', 'max-pending-requests', 'pending-timeout-ms',
             'default-max-tokens', 'device'],
    'kv': ['kv-dtype', 'kv-layer-storage', 'kv-bit-budget', 'kv-residual-layers',
           # kv-rotation/kv-row-scale/kv-tier-formats/kv-v-codec exist in the
           # live engine's serve_options.cpp but sit in no group here (they were
           # found by the parse chain only), so a regeneration filed them under
           # 'ungrouped'.  Added 2026-09-13 with the i18n port.
           'kv-rotation', 'kv-row-scale', 'kv-tier-formats', 'kv-v-codec',
           # The K/V bit-width entries and the score table's own entry point
           # (product/kv_kv_bits.h), plus the two-score slider. Added with the
           # line-6 K/V work; without them a regeneration files them as 'ungrouped'.
           'kv-bits', 'kv-k-bits', 'kv-v-bits', 'kv-bits-mode',
           'kv-quality-weight', 'kv-tier-scores', 'kv-k-tier-scores', 'kv-v-tier-scores',
           'kv-score-table',
           'cold-policy', 'cold-keep-tokens', 'max-cold-pages', 'cold-host-bytes',
           'cold-disk-path', 'cold-disk-bytes', 'weight-host-bytes', 'host-kv-mib',
           'host-state-slots', 'device-state-slots'],
    'speculative': ['spec', 'draft-tokens', 'lm-head-draft'],
    'sampling': ['temperature', 'top-p', 'top-k', 'min-p', 'presence-penalty',
                 'frequency-penalty', 'seed', 'greedy'],
    'thinking': ['no-thinking', 'default-thinking-budget', 'preserve-thinking'],
    'media': ['vision', 'media-cache-mib', 'media-live-mib', 'media-preprocess-threads'],
    # merged 2026-09-13 from the live tree's tools/gui/extract_serve_params.py:
    # this list alone did not know about these four flags, so a regeneration
    # filed them under 'ungrouped' (the repository GUI reads this file).
    'perf': ['no-cuda-graph', 'graph-capture-ceiling', 'prefill-chunk', 'no-prefix-reuse',
             'no-auto-system-shared-prefix'],
    'network': ['cors', 'max-request-mib'],
    'diagnostics': ['log-stats-interval-ms', 'request-log-jsonl', 'raw-output',
                    'print-token-ids', 'reasoning-effort'],
    'advanced': ['context-cost-presets', 'response-store-max-records',
                 'response-store-max-mib', 'max-private-continuations',
                 'max-shared-prefixes', 'max-long-anchors-per-continuation', 'yarn'],
}
GROUP_ORDER = tuple(GROUPS) + ('ungrouped',)

# 引擎里"是旗标但不是服务参数"的东西, 抽取时保留, GUI 侧自行决定是否渲染.
_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
# 注意 '--' 必须放在捕获组之外, 否则 name 会带上前导横线, flag 变成 "----xxx".
_ARG_RE = re.compile(r'arg\s*==\s*"--([a-z0-9][a-z0-9-]*)"')


def resolve_src(cli_src: str | None = None) -> Path:
    """--src > $NINFER_SERVE_SRC > DEFAULT_SRC."""
    for cand in (cli_src, os.environ.get('NINFER_SERVE_SRC')):
        if cand:
            return Path(cand).expanduser()
    return Path(DEFAULT_SRC)


def _usage_text(text: str) -> str:
    """把 serve_usage_text() 里拼接的各段字面量还原成一段纯文本(保留换行, 保住行结构).

    C++ 侧是 `"literal" + std::to_string(x) + "literal"`, 非字面量的代码段用 '…'
    占位, 这样 "defaults to N when omitted" 仍能读通.
    """
    i = text.find('"usage: ')
    if i < 0:
        return ''
    j = text.find(';\n', i)
    chunk = text[i:j if j > 0 else len(text)]
    out = []
    prev_end = None
    for m in _LITERAL_RE.finditer(chunk):
        if prev_end is not None:
            gap = chunk[prev_end:m.start()]
            if re.sub(r'[\s+]', '', gap):
                out.append('…')
        out.append(m.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\'))
        prev_end = m.end()
    return ''.join(out)


def _brace_body(text: str, start: int) -> str:
    """从 start 之后第一个 '{' 起, 用配对花括号取出一段 body(比非贪婪正则可靠)."""
    i = text.find('{', start)
    if i < 0:
        return ''
    depth = 0
    for k in range(i, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:k]
    return text[i + 1:]


def _usage_hints(usage: str) -> tuple[dict, dict]:
    """从 usage 文本里取 (flag -> 值提示 token) 和 (flag -> 说明文字).

    usage 的排版约定: 第一行是 "[--flag TOKEN] [--other TOKEN]" 摘要,
    之后每行是一条以 "--flag 说明…" 开头的散文注释 —— 按行解析即可.
    """
    args: dict[str, str] = {}
    notes: dict[str, str] = {}

    # a) [--flag TOKEN --other TOKEN] 方括号组: 按 -- 切块, 块头是旗标, 余下是值提示.
    for grp in re.findall(r'\[([^\[\]]*--[^\[\]]*)\]', usage):
        for chunk in re.split(r'(?=--[a-z0-9-])', grp):
            chunk = chunk.strip()
            if not chunk.startswith('--'):
                continue
            name, _, rest = chunk.partition(' ')
            args.setdefault(name[2:].strip(), rest.strip())

    # b) 注释行 "--flag 说明…" => hint.
    for line in usage.split('\n'):
        s = line.strip()
        m = re.match(r'--([a-z0-9-]+)\s+(\S.*)$', s)
        if m:
            notes.setdefault(m.group(1), m.group(2).strip())
    return args, notes


def _parse_chain(text: str) -> dict:
    """扫 parse 链: arg == "--xxx" { body } => 是否带值 + 值类型 + 枚举候选."""
    out: dict[str, dict] = {}
    for m in _ARG_RE.finditer(text):
        name = m.group(1)
        body = _brace_body(text, m.end())
        if not body:
            continue
        has_value = 'require_value' in body
        if re.search(r'parse_bool', body):
            vtype = 'bool'
        elif re.search(r'parse_float_in|parse_float|parse_f32|parse_double', body):
            vtype = 'float'
        elif re.search(r'parse_u64|parse_u32|parse_nonnegative_int|parse_int', body):
            # 逗号分隔的整数表(如 --kv-residual-layers)按字符串收, 但标记出来.
            vtype = 'string' if re.search(r"""find\(\s*','|find\(\s*","|split\(\s*','""", body) else 'int'
        elif not has_value:
            vtype = 'bool'
        else:
            vtype = 'string'
        choices = []
        if vtype == 'string':
            for c in re.findall(r'==\s*"([A-Za-z0-9_.|+-]+)"', body):
                if c not in choices:
                    choices.append(c)
        out[name] = {'takes_value': vtype != 'bool', 'type': vtype, 'choices': choices,
                     'list': bool(re.search(r"""find\(\s*','|find\(\s*","|split\(\s*','""", body))}
    return out


def extract(src: Path | None = None) -> dict:
    src = src or resolve_src()
    text = src.read_text(encoding='utf-8', errors='replace')
    usage = _usage_text(text)
    uargs, unotes = _usage_hints(usage)
    chain = _parse_chain(text)

    flags: dict[str, dict] = {}

    # 1) parse 链是权威来源: 每条都登记.
    for name, info in sorted(chain.items()):
        flags[name] = {
            'flag': '--' + name,
            'arg': uargs.get(name, ''),
            'takes_value': info['takes_value'],
            'type': info['type'],
        }
        if info['choices']:
            flags[name]['choices'] = info['choices']
        if info.get('list'):
            flags[name]['list'] = True

    # 2) usage 里出现但 parse 链没覆盖的旗标(例如 --help)照样登记, 类型靠值提示猜.
    for name, tok in sorted(uargs.items()):
        if name in flags:
            continue
        tok = tok.split()[0] if tok else ''
        flags[name] = {
            'flag': '--' + name,
            'arg': tok,
            'takes_value': False,
            'type': 'bool',
        }

    # 3) usage 的枚举提示 "a|b|c" => choices; 同时把缺的 arg 补上.
    for name, f in flags.items():
        tok = uargs.get(name, '')
        if tok and not f.get('arg'):
            f['arg'] = tok
        if '|' in f.get('arg', '') and not f.get('choices'):
            f['choices'] = f['arg'].split('|')

    # 4) 说明文字 / 默认值说明.
    for name, f in flags.items():
        note = unotes.get(name, '')
        if note:
            f['hint'] = note
            d = re.search(r'defaults to (…|[^.;]+)', note)
            if d:
                # 说明句里 "384 and is enforced …" 只取数字部分.
                f['default'] = re.split(r'\s+(?:and|when)\s+|;', d.group(1))[0].strip()

    # 5) 分组.
    grouped: dict[str, list] = {g: [] for g in GROUPS}
    ungrouped = []
    for name in sorted(flags):
        for g, names in GROUPS.items():
            if name in names:
                grouped[g].append(flags[name])
                break
        else:
            ungrouped.append(flags[name])

    return {
        'source': str(src),
        'source_sha1': hashlib.sha1(text.encode('utf-8', 'replace')).hexdigest(),
        'total': len(flags),
        'parse_chain_flags': len(chain),
        'group_order': list(GROUP_ORDER),
        'groups': grouped,
        'ungrouped': ungrouped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'serve_params.json'))
    ap.add_argument('--src', default='', help='serve_options.cpp 路径 (默认 $NINFER_SERVE_SRC 或内置默认)')
    a = ap.parse_args()
    src = resolve_src(a.src or None)
    if not src.exists():
        print('source not found: %s' % src, file=sys.stderr)
        print('hint: --src PATH 或 NINFER_SERVE_SRC=PATH', file=sys.stderr)
        return 2
    reg = extract(src)
    Path(a.out).write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding='utf-8')
    print('source:', reg['source'])
    print('sha1  :', reg['source_sha1'][:16])
    print('total flags:', reg['total'], '(parse chain %d)' % reg['parse_chain_flags'])
    for g in GROUP_ORDER:
        items = reg['groups'].get(g, []) if g != 'ungrouped' else reg['ungrouped']
        print(' %-12s %2d' % (g, len(items)), [i['flag'] for i in items][:6])
    print('written:', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
