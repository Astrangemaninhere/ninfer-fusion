#!/bin/bash
# 修掉 ssv2.py 里过严的断言（JSON 是 '{\n' 换行格式，不是 '{"'）
python3 - <<'PY'
import pathlib
p = pathlib.Path('/home/user/ssv2.py')
s = p.read_text(encoding='utf-8', errors='surrogateescape')
old_a = 'assert head.lstrip()[:2] == b\'{"\''
new_a = 'assert head.lstrip()[:1] == b"{"'
if old_a in s:
    s = s.replace(old_a, new_a)
    p.write_text(s, encoding='utf-8', errors='surrogateescape')
    print('patched OK')
else:
    # 退路：把含 assert 的整行替换
    lines = s.splitlines(True)
    out = []
    done = False
    for ln in lines:
        if (not done) and 'assert' in ln and 'head' in ln:
            out.append('assert head.lstrip()[:1] == b"{", "payload start wrong"\n')
            done = True
        else:
            out.append(ln)
    p.write_text(''.join(out), encoding='utf-8', errors='surrogateescape')
    print('patched by line:', done)
PY
python3 /home/user/ssv2.py 2>&1 | head -40
