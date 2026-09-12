#!/usr/bin/env python3
"""在 vllm venv 里用 safetensors 直接打开目标分片，复现完整报错。"""
import json, os, struct, traceback

T = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-NVFP4-RTX5090"
files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

print("=== 1) 头部的 __metadata__（部分加载器要求 format=pt）===")
for fn in files:
    p = os.path.join(T, fn)
    with open(p, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hn).decode("utf-8"))
    md = hdr.get("__metadata__", None)
    print("  %-34s __metadata__ = %s  张量数=%d" % (fn, md, len([k for k in hdr if k != '__metadata__'])))

print("\n=== 2) 用 safetensors.safe_open 打开（复现 vLLM 的行为）===")
try:
    from safetensors import safe_open
    for fn in files:
        p = os.path.join(T, fn)
        try:
            with safe_open(p, framework="pt") as f:
                keys = list(f.keys())
                print("  %-34s OK  张量数=%d  例: %s" % (fn, len(keys), keys[0]))
        except Exception as e:
            print("  %-34s FAIL: %s: %s" % (fn, type(e).__name__, e))
except ImportError as e:
    print("  safetensors 未安装:", e)

print("\n=== 3) 草稿目录（vLLM 可能扫到 .bak 那个坏文件）===")
D = r"C:\Users\User\Documents\ziqinzhang\data\draft_model"
for fn in sorted(os.listdir(D)):
    p = os.path.join(D, fn)
    if not os.path.isfile(p):
        continue
    print("  %-46s %12d" % (fn, os.path.getsize(p)))
    if "safetensors" in fn:
        try:
            from safetensors import safe_open
            with safe_open(p, framework="pt") as f:
                print("       -> safe_open OK, 张量数=%d" % len(list(f.keys())))
        except Exception as e:
            print("       -> safe_open FAIL: %s: %s" % (type(e).__name__, e))
