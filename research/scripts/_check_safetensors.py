#!/usr/bin/env python3
"""校验 safetensors 文件完整性：头部长度是否与文件大小自洽（截断检测）。"""
import json, os, struct, pathlib

cands = []
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/data/draft_model")
for p in sorted(T.glob("*.safetensors")): cands.append(p)
for p in sorted(D.iterdir()):
    if p.is_file() and ("safetensors" in p.name): cands.append(p)

print("%-64s %14s %14s %10s %s" % ("文件", "实际大小", "头部声明需", "差值", "判定"))
for p in cands:
    size = os.path.getsize(p)
    try:
        with open(p, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            if hn <= 0 or hn > 200 * 1024 * 1024:
                print("%-64s %14d  头部长度异常: %d" % (p.name, size, hn)); continue
            raw = f.read(hn)
        try:
            hdr = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print("%-64s %14d  头部 JSON 解析失败: %s" % (p.name, size, e)); continue
        need = 8 + hn
        mx = 0
        for k, v in hdr.items():
            if k == "__metadata__": continue
            off = v.get("data_offsets")
            if off: mx = max(mx, off[1])
        need_total = 8 + hn + mx
        ok = (size >= need_total) and (size - need_total < 4096)
        print("%-64s %14d %14d %10d %s" % (
            p.name, size, need_total, size - need_total,
            "OK" if ok else ("截断(缺 %d 字节)" % (need_total - size) if size < need_total else "多余 %d" % (size - need_total))))
    except Exception as e:
        print("%-64s %14d  读取失败: %s" % (p.name, size, e))

print("\n=== 其它可能被误加载的文件 ===")
for p in sorted(D.iterdir()):
    print("  %-52s %12d" % (p.name, os.path.getsize(p)))
print("\n=== 目标目录全部文件 ===")
for p in sorted(T.iterdir()):
    if p.is_file():
        print("  %-52s %12d" % (p.name, os.path.getsize(p)))
