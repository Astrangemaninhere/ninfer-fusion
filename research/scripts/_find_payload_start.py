#!/usr/bin/env python3
"""定位 .ninfer 载荷区真实起点：头部结束(16+json_len)之后，扫第一个"长可打印串"。"""
import os, struct
P = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
with open(P, "rb") as f:
    magic = f.read(8); n = struct.unpack("<Q", f.read(8))[0]
hdr_end = 16 + n
print("magic=%s json_len=%d hdr_end=%d" % (magic.hex(), n, hdr_end))

def printable_run(b, start, need=200):
    run = 0
    for i, c in enumerate(b):
        if 32 <= c < 127:
            run += 1
            if run >= need:
                return start + i - need + 1
        else:
            run = 0
    return -1

with open(P, "rb") as f:
    # 从头部结束前 4 字节开始读 2 MB
    f.seek(max(0, hdr_end - 4))
    buf = f.read(2 * 1024 * 1024)
base = max(0, hdr_end - 4)
pos = printable_run(buf, base)
print("载荷区内第一个长可打印串起点 = %s（相对 hdr_end = %s）" % (pos, pos - hdr_end if pos >= 0 else None))

# 对齐候选
for al in (16, 64, 256, 4096, 65536, 1 << 20):
    cand = ((hdr_end + al - 1) // al) * al
    if pos >= 0 and abs(pos - cand) < 4096:
        print("  -> 与 %d 字节对齐吻合（候选 %d, 差 %d）" % (al, cand, pos - cand))

# 直接看几个候选处的字节
print("\n各候选起点的前 32 字节:")
with open(P, "rb") as f:
    for al in (1, 16, 64, 256, 4096, 65536, 1 << 20):
        cand = ((hdr_end + al - 1) // al) * al
        f.seek(cand)
        print("   align=%-8d @%-12d %r" % (al, cand, f.read(32)))
