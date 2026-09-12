#!/usr/bin/env python3
"""决定性比对：参照草稿 incoai/Qwen3.8-27B-DFlash2 的权重
vs 我们产物里内嵌的 dflash2/ 草稿（qwen3_8_27b_nvfp4_dflash2.ninfer）。
两者同名同形（去 dflash2/ 前缀即 HF 名），且都是 BF16 ⇒ 可逐字节比。
若不同 ⇒ 我们内嵌的草稿不是这份参照草稿（接受率崩的原因）。"""
import json, pathlib, struct, hashlib

NINFER = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
REF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/data/draft_dflash2_ref")

def align_up(x, a): return ((x + a - 1) // a) * a

# --- 读 ninfer 的 dflash2/ 张量 ---
with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
PAY = align_up(16 + n, 4096)
d2 = {o["name"]: o for o in man["objects"]
      if o.get("kind") == "tensor" and o["name"].startswith("dflash2/")}
print("产物 dflash2/ 张量: %d 个" % len(d2))
print("产物 dflash2/ 全部名字（前 12）:")
for nm in sorted(d2)[:12]:
    print("   %s" % nm)

# --- 读参照草稿的 safetensors 头 ---
sf = REF / "model.safetensors"
if not sf.exists():
    print("\n参照草稿尚未下载完: %s" % sf); raise SystemExit(0)
with open(sf, "rb") as f:
    hn = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hn).decode("utf-8"))
ref_names = [k for k in hdr if k != "__metadata__"]
print("\n参照草稿张量: %d 个" % len(ref_names))
print("参照草稿全部名字（前 12）:")
for nm in sorted(ref_names)[:12]:
    print("   %s" % nm)

def read_ref(name):
    e = hdr.get(name)
    if not e: return None
    with open(sf, "rb") as f:
        f.seek(8 + hn + e["data_offsets"][0])
        return f.read(e["data_offsets"][1] - e["data_offsets"][0])

def read_nin(name):
    o = d2[name]
    with open(NINFER, "rb") as f:
        f.seek(PAY + o["offset"]); return f.read(o["bytes"])

# --- 逐同名比对（去 dflash2/ 前缀）---
same = diff = miss = 0
examples = []
for nm in sorted(d2):
    short = nm[len("dflash2/"):]
    b = read_ref(short)
    if b is None:
        miss += 1; continue
    a = read_nin(nm)
    if len(a) != len(b):
        diff += 1
        if len(examples) < 4: examples.append((short, "长度 %d vs %d" % (len(a), len(b))))
        continue
    if a == b:
        same += 1
    else:
        diff += 1
        if len(examples) < 4:
            examples.append((short, "md5 %s vs %s" % (hashlib.md5(a).hexdigest()[:10],
                                                      hashlib.md5(b).hexdigest()[:10])))

print("\n=== 判定 ===")
print("  同名逐字节一致: %d   不同: %d   参照里没有: %d" % (same, diff, miss))
for nm, why in examples:
    print("   例: %-46s %s" % (nm, why))
if same > 0 and diff == 0:
    print("  => 我们内嵌的草稿【就是】这份参照草稿 ⇒ 草稿不是原因，问题在引擎侧输入路径")
elif diff > 0:
    print("  => 我们内嵌的草稿【不是】这份参照草稿 ⇒ 这就是接受率崩的原因（换了坏草稿）")
