#!/usr/bin/env python3
"""带 spec 配置 + --enforce-eager（跳过 CUDA graph 捕获，绕开挂点）的变体。
从 _vllm_ref3.py 生成，只加一个旗标。"""
import pathlib, re
src = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_vllm_ref3.py")
s = src.read_text(encoding="utf-8", errors="surrogateescape")
old = '"--max-num-seqs", "32",'
new = '"--max-num-seqs", "32",\n        "--enforce-eager",'
if old in s and "--enforce-eager" not in s:
    s = s.replace(old, new, 1)
    print("added --enforce-eager")
else:
    print("pattern state: has_eager=%s has_maxseq=%s" % ("--enforce-eager" in s, old in s))
# 换个端口避免与存活实例冲突
s = s.replace("PORT = 8125", "PORT = 8126")
pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_vllm_spec_eager.py").write_text(
    s, encoding="utf-8", errors="surrogateescape")
print("written _vllm_spec_eager.py (port 8126)")
# 打印 args 段确认
for ln in s.splitlines():
    if "max-num-seqs" in ln or "enforce-eager" in ln or "speculative-config" in ln or ln.strip().startswith("PORT"):
        print("   " + ln.strip())
