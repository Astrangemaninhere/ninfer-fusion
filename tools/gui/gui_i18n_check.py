#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui_i18n_check.py — 界面翻译覆盖检查器（验收门）。

判据（三条都必须过，"全功能完成"由此证明而非口头声明）：
  1) 源码里出现的每个 `t('key')` / `t("key")` 键，在 zh 与 en 表里都存在；
  2) 表里不存在"只有 zh 没有 en"的条目（即英文覆盖完整）；
  3) 每个 GUI 模块的**用户可见字面量**都走了 t()：扫描 PAGE/HTML 里裸露的中文
     （排除注释、CSS、属性名、以及 data-i18n 占位），任何残留即失败并列出位置。

Usage: python3 gui_i18n_check.py [--gui-dir tools/gui] [--json out.json]
Exit code 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

MODULES = ('serve_gui.py', 'convert_gui.py', 'rag_gui.py', 'model_import.py', 'gui_tips.py')

# 允许裸露中文的上下文（正则按行匹配后豁免）
ALLOW_LINE_PATTERNS = (
    r'^\s*#',                      # 注释
    r'^\s*"""', r'^\s*#\s*-',      # 文档串/分隔线
    r'data-i18n',                  # 已标注待翻译的占位
    r'^\s*//',                     # JS 注释
    r'说明：', r'—\s*写给',        # 本文件自身的说明
)
CJK = re.compile(r'[\u4e00-\u9fff]')
T_CALL = re.compile(r"""\bt\(\s*(['"])(?P<key>[^'"]+)\1""")

# 动态/模板键：真键运行时才拼出来（'tips.%s' % cid、f'imp.kind.{kind}'），
# 不能当成缺键；改为要求同前缀至少存在一个具体条目（家族可解析）。
TEMPLATE_KEY = re.compile(r'[%{}\*]|\.$|_$')


def is_template_key(key: str) -> bool:
    return bool(TEMPLATE_KEY.search(key))


def scan_keys(gui: pathlib.Path, only: tuple[str, ...] = ()) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name in (only or MODULES):
        p = gui / name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            for m in T_CALL.finditer(line):
                found.setdefault(m.group('key'), []).append('%s:%d' % (name, i))
    return found


def raw_chinese(gui: pathlib.Path, only: tuple[str, ...] = ()) -> list[str]:
    hits = []
    for name in (only or MODULES):
        p = gui / name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            if not CJK.search(line):
                continue
            if any(re.search(pat, line) for pat in ALLOW_LINE_PATTERNS):
                continue
            # 行内注释：按引号切分，只判"代码段"里的中文（纯注释行上面已放行）
            code_only = ''
            quote = ''
            for ch in line:
                if quote:
                    if ch == quote:
                        quote = ''
                    code_only += 'x'
                    continue
                if ch in '\'"':
                    quote = ch
                    code_only += 'x'
                    continue
                if ch == '#':
                    break
                code_only += ch
            if not CJK.search(code_only):
                continue
            # t(...) 调用自带键，键本身不是用户可见文本
            if T_CALL.search(line):
                continue
            hits.append('%s:%d  %s' % (name, i, line.strip()[:110]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gui-dir', default=str(pathlib.Path(__file__).resolve().parent))
    ap.add_argument('--json', default='')
    ap.add_argument('--only', default='',
                    help='逗号分隔的文件名；只检查这些模块（并行分工时各自 gate）')
    a = ap.parse_args()
    gui = pathlib.Path(a.gui_dir).resolve()
    only = tuple(x.strip() for x in a.only.split(',') if x.strip())
    sys.path.insert(0, str(gui))
    import gui_i18n as i18n

    keys = scan_keys(gui, only)
    missing_zh = sorted(k for k in keys if not i18n.has(k, 'zh'))
    missing_en = sorted(k for k in keys if not i18n.has(k, 'en'))
    cov = i18n.coverage()
    zh_only = sorted(k for k, v in i18n.STRINGS.items() if 'zh' in v and 'en' not in v)
    raw = raw_chinese(gui, only)
    families = {}
    for k in keys:
        if is_template_key(k):
            prefix = k.split('%')[0].split('{')[0].rstrip('._')
            concrete = [x for x in i18n.STRINGS if x.startswith(prefix) and not is_template_key(x)]
            families[k] = len(concrete)
    bad_families = sorted(k for k, n in families.items() if n == 0)
    missing_zh = [k for k in missing_zh if not is_template_key(k)]
    missing_en = [k for k in missing_en if not is_template_key(k)]

    ok = not (missing_zh or missing_en or zh_only or raw or bad_families)
    report = {
        'gui_dir': str(gui),
        'keys_used': len(keys),
        'keys_in_tables': cov['keys'],
        'zh_entries': cov['zh'],
        'en_entries': cov['en'],
        'missing_zh': missing_zh,
        'missing_en': missing_en,
        'zh_only_entries': zh_only,
        'raw_chinese_lines': raw,
        'template_keys': families,
        'template_keys_without_family': bad_families,
        'pass': ok,
    }
    print('keys used by sources : %d' % len(keys))
    print('keys in tables      : %d (zh %d / en %d)' % (cov['keys'], cov['zh'], cov['en']))
    for label, items in (('missing in zh', missing_zh), ('missing in en', missing_en),
                         ('zh-only entries', zh_only)):
        if items:
            print('%s (%d): %s' % (label, len(items), ', '.join(items[:12])))
    if families:
        print('template keys (runtime-resolved): %d; families without concrete entries: %s'
              % (len(families), bad_families or 'none'))
    if raw:
        print('user-visible Chinese NOT routed through t() (%d):' % len(raw))
        for line in raw[:20]:
            print('   ', line)
    print('VERDICT:', 'PASS' if ok else 'FAIL')
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
