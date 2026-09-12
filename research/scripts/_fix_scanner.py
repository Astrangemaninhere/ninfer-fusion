#!/usr/bin/env python3
"""修扫描器的花括号记账（`namespace { ... }` 单行自闭合会被误判为"仍在命名空间内"），
避免以后把无害写法当 bug 去"修"。"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_scan_include_in_ns.py")
s = P.read_text(encoding="utf-8")
s = s.replace("            depth += s.count(\"{\")",
              "            # 单行 `namespace { ... }` 自闭合：两个都要算，否则会假阳性\n"
              "            depth += s.count(\"{\") - s.count(\"}\")")
s = s.replace('        if s.startswith("#include") and first_ns is not None and depth > 0:',
              '        if s.startswith("#include") and first_ns is not None and depth > 0:')
P.write_text(s, encoding="utf-8")
print("scanner 已修（花括号记账）")
