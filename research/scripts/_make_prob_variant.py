#!/usr/bin/env python3
"""把 vLLM 的 dspark 草稿采样从 greedy 改成 probabilistic（参照实测：16%→28%）。
生成 _vllm_clean_prob.py（端口 8128）。"""
import pathlib
src = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_vllm_clean.py")
s = src.read_text(encoding="utf-8", errors="surrogateescape")
if "draft_sample_method" not in s:
    print("找不到 draft_sample_method，先看文件里的 spec 构造")
    for ln in s.splitlines():
        if "spec" in ln and "=" in ln:
            print("   " + ln.strip()[:150])
else:
    s2 = s.replace('"draft_sample_method": "greedy"', '"draft_sample_method": "probabilistic"')
    s2 = s2.replace('"draft_sample_method":"greedy"', '"draft_sample_method":"probabilistic"')
    s2 = s2.replace("PORT = 8127", "PORT = 8128")
    pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_vllm_clean_prob.py").write_text(
        s2, encoding="utf-8", errors="surrogateescape")
    print("written _vllm_clean_prob.py (port 8128)")
    for ln in s2.splitlines():
        if "draft_sample_method" in ln or ln.strip().startswith("PORT"):
            print("   " + ln.strip()[:150])
