#!/usr/bin/env python3
"""修检查器的两处假阳性（用行内插入，锚点按文件实际文本核对）：
  1) 动态/模板键（'tips.%s'、f'imp.kind.{x}'）不算缺键，改为检查"家族有具体条目"；
  2) 行内 `# 注释` 里的中文不算裸露中文。"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui/gui_i18n_check.py")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)

# --- 1) T_CALL 之后插入模板键判定 ---
anchor = 'T_CALL = re.compile(r"""\\bt\\(\\s*([\'"])(?P<key>[^\'"]+)\\1""")\n'
idx = [i for i, l in enumerate(lines) if l.startswith('T_CALL = re.compile(')]
assert len(idx) == 1, "T_CALL 锚点不唯一: %d" % len(idx)
helper = (
    "\n# 动态/模板键：真键运行时才拼出来（'tips.%s' % cid、f'imp.kind.{kind}'），\n"
    "# 不能当成缺键；改为要求同前缀至少存在一个具体条目（家族可解析）。\n"
    "TEMPLATE_KEY = re.compile(r'[%{}\\*]|\\.$|_$')\n"
    "\n"
    "\n"
    "def is_template_key(key: str) -> bool:\n"
    "    return bool(TEMPLATE_KEY.search(key))\n"
)
lines.insert(idx[0] + 1, helper)

text = "".join(lines)

# --- 2) raw_chinese：行内注释不算裸露中文 ---
old = """            if any(re.search(pat, line) for pat in ALLOW_LINE_PATTERNS):
                continue
            # t(...) 调用自带键，键本身不是用户可见文本"""
new = """            if any(re.search(pat, line) for pat in ALLOW_LINE_PATTERNS):
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
                if ch in '\\'\"':
                    quote = ch
                    code_only += 'x'
                    continue
                if ch == '#':
                    break
                code_only += ch
            if not CJK.search(code_only):
                continue
            # t(...) 调用自带键，键本身不是用户可见文本"""
assert text.count(old) == 1, "raw_chinese 锚点不唯一"
text = text.replace(old, new)

# --- 3) main(): 模板键家族 + 不再把模板键算缺键 ---
old2 = """    raw = raw_chinese(gui, only)"""
new2 = """    raw = raw_chinese(gui, only)
    families = {}
    for k in keys:
        if is_template_key(k):
            prefix = k.split('%')[0].split('{')[0].rstrip('._')
            concrete = [x for x in i18n.STRINGS if x.startswith(prefix) and not is_template_key(x)]
            families[k] = len(concrete)
    bad_families = sorted(k for k, n in families.items() if n == 0)
    missing_zh = [k for k in missing_zh if not is_template_key(k)]
    missing_en = [k for k in missing_en if not is_template_key(k)]"""
assert text.count(old2) == 1, "raw 锚点不唯一"
text = text.replace(old2, new2)

old3 = """    ok = not (missing_zh or missing_en or zh_only or raw)"""
new3 = """    ok = not (missing_zh or missing_en or zh_only or raw or bad_families)"""
assert text.count(old3) == 1
text = text.replace(old3, new3)

old4 = """        'raw_chinese_lines': raw,
        'pass': ok,"""
new4 = """        'raw_chinese_lines': raw,
        'template_keys': families,
        'template_keys_without_family': bad_families,
        'pass': ok,"""
assert text.count(old4) == 1
text = text.replace(old4, new4)

old5 = """    if raw:
        print('user-visible Chinese NOT routed through t()"""
new5 = """    if families:
        print('template keys (runtime-resolved): %d; families without concrete entries: %s'
              % (len(families), bad_families or 'none'))
    if raw:
        print('user-visible Chinese NOT routed through t()"""
assert text.count(old5) == 1
text = text.replace(old5, new5)

P.write_text(text, encoding="utf-8")
print("checker patched: %d lines" % (text.count("\n") + 1))
