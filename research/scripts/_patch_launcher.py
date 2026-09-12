#!/usr/bin/env python3
"""收紧 vLLM 启动器的错误检测正则（原正则含裸 'error:'，会误判 c10d socket 警告）。"""
import pathlib
p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_vllm_launch.py")
s = p.read_text(encoding="utf-8", errors="surrogateescape")
old = 'if re.search(r"ModuleNotFoundError|ValueError|RuntimeError|Traceback|error:|not supported|No module", t):'
new = 'if re.search(r"Traceback \\(most recent call last\\)|ModuleNotFoundError|ImportError|api_server\\.py: error:|^ValueError: |^RuntimeError: ", t):'
if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", errors="surrogateescape")
    print("regex fixed")
else:
    print("old pattern not found; current re.search lines:")
    for ln in s.splitlines():
        if "re.search" in ln:
            print("  " + ln.strip())
