#!/usr/bin/env python3
"""adapt.py 缺 `import re`（我的新探测器用到），补上。"""
import pathlib
import re as _re

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
src = P.read_text(encoding="utf-8")
if _re.search(r"^import re$", src, _re.M):
    print("已有 import re")
else:
    m = _re.search(r"^import os$", src, _re.M)
    assert m, "找不到 import os 锚点"
    src = src[:m.end()] + "\nimport re" + src[m.end():]
    P.write_text(src, encoding="utf-8")
    print("已插入 import re")
print("imports:", [l for l in src.splitlines()[:30] if l.startswith("import ")][:8])
