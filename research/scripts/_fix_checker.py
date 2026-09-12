#!/usr/bin/env python3
"""两个假阳性修掉，让检查器对"动态键"和"行内注释"判得准：
  1) `t('tips.%s' % cid)` / `t(f'imp.kind.{x}')` 这类**模板键**不是真键：跳过含 %/{}/ 或
     以 . 或 _ 结尾的写法，改为要求"同前缀至少有一个具体键"（家族可解析）；
  2) 行内 `# 注释`（代码后面的中文注释）不应算裸露中文。
"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui/gui_i18n_check.py")
src = P.read_text(encoding="utf-8")

A = """CJK = re.compile(r'[\\u4e00-\\u9fff]')
T_CALL = re.compile(r\"\"\"\\bt\\(\\s*(['\\\"])(?P<key>[^'\\\"]+)\\1\"\"\")"""
B = """CJK = re.compile(r'[\\u4e00-\\u9fff]')
T_CALL = re.compile(r\"\"\"\\bt\\(\\s*(['\\\"])(?P<key>[^'\\\"]+)\\1\"\"\")
# 模板/动态键：真键在运行时拼出来（'tips.%s' % cid、f'imp.kind.{kind}'），
# 检查器不能把它们当缺键，改为检查"家族有具体条目"。
TEMPLATE_KEY = re.compile(r'[%{}\\*]|\\.$|_$')


def is_template_key(key: str) -> bool:
    return bool(TEMPLATE_KEY.search(key))"""
assert src.count(A) == 1, "CJK/T_CALL anchor not unique"
src = src.replace(A, B)

C = """    raw = raw_chinese(gui, only)"""
D = """    raw = raw_chinese(gui, only)
    # 模板键只要求家族可解析（同前缀至少一个具体键在表里）
    families = {}
    for k in keys:
        if is_template_key(k):
            prefix = k.split('%')[0].split('{')[0].rstrip('._')
            concrete = [x for x in i18n.STRINGS if x.startswith(prefix) and not is_template_key(x)]
            families[k] = len(concrete)
    bad_families = sorted(k for k, n in families.items() if n == 0)
    missing_zh = [k for k in missing_zh if not is_template_key(k)]
    missing_en = [k for k in missing_en if not is_template_key(k)]"""
assert src.count(C) == 1, "raw anchor not unique"
src = src.replace(C, D)

E = """    ok = not (missing_zh or missing_en or zh_only or raw)"""
F = """    ok = not (missing_zh or missing_en or zh_only or raw or bad_families)"""
assert src.count(E) == 1
src = src.replace(E, F)

G = """        'raw_chinese_lines': raw,
        'pass': ok,"""
H = """        'raw_chinese_lines': raw,
        'template_keys': families,
        'template_keys_without_family': bad_families,
        'pass': ok,"""
assert src.count(G) == 1
src = src.replace(G, H)

I = """    if raw:"""
J = """    if families:
        print('template keys (resolved at runtime): %d, families without concrete entries: %s'
              % (len(families), bad_families or 'none'))
    if raw:"""
assert src.count(I) == 1
src = src.replace(I, J)

# 行内注释：把 '#' 之后（且不在引号内）的部分裁掉再判中文
K = """            if any(re.search(pat, line) for pat in ALLOW_LINE_PATTERNS):
                continue"""
L = """            if any(re.search(pat, line) for pat in ALLOW_LINE_PATTERNS):
                continue
            # 行内注释：按引号切分，只检查"代码段"里的中文；纯注释行已在上面放行
            code_only = ''
            quote = ''
            for ch in line:
                if quote:
                    if ch == quote:
                        quote = ''
                    code_only += 'x'
                    continue
                if ch in '\\'\"':
                    quote = ch
                    code_only += 'x'
                    continue
                if ch == '#':
                    break
                code_only += ch
            if not CJK.search(code_only):
                continue"""
assert src.count(K) == 1
src = src.replace(K, L)

P.write_text(src, encoding="utf-8")
print("checker updated: template-key handling + inline-comment awareness (%d lines)"
      % (src.count("\n") + 1))
