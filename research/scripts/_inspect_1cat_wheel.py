#!/usr/bin/env python3
"""检查 1Cat wheel 的内容与依赖（判断在 SM120 上能否真跑）。"""
import zipfile
W = "/mnt/c/Users/User/Documents/ziqinzhang/dl/1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl"
z = zipfile.ZipFile(W)
names = z.namelist()
print("条目数: %d" % len(names))
so = [n for n in names if n.endswith((".so", ".ptx", ".cubin"))]
print("二进制条目: %d" % len(so))
for n in so[:30]:
    print("   %-70s %d B" % (n, z.getinfo(n).file_size))
print()
meta = [n for n in names if n.endswith("METADATA")]
if meta:
    txt = z.read(meta[0]).decode("utf-8", "replace")
    print("=== METADATA ===")
    for line in txt.splitlines():
        if line.startswith(("Name:", "Version:", "Requires-Python", "Requires-Dist")):
            print("   " + line[:130])
print()
# 找 dflash2 / lookup 是否在 wheel 里
print("=== 关键模块是否打包 ===")
for pat in ("dflash2", "lookup", "ngram", "qwen3_dflash", "dflash_sm70", "flash_attn_v100", "spec_decode"):
    hits = [n for n in names if pat in n]
    print("   %-18s %d 个" % (pat, len(hits)))
