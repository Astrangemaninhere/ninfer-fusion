#!/usr/bin/env python3
"""去掉 _vllm_ref3.py 里的自杀式清理块（清理改由 PowerShell 在启动前做）。"""
import pathlib
p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_vllm_ref3.py")
s = p.read_text(encoding="utf-8", errors="surrogateescape")

start = s.index("# --- 1) 清残留 ---")
end = s.index("spec = {")
block = s[start:end]
replacement = "# --- 1) 清残留：改由调用方在启动前完成（避免自杀）---\n"
s2 = s[:start] + replacement + s[end:]
p.write_text(s2, encoding="utf-8", errors="surrogateescape")
print("removed %d chars of self-killing cleanup" % len(block))
print("--- new head ---")
print("\n".join(s2.splitlines()[:14]))
