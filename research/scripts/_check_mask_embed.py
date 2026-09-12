#!/usr/bin/env python3
"""检验：草稿第 1..K 位吃的 mask token（248070）在 embedding 表里是不是"从未训练"的行。

契约 §6.1：草稿没有自己的 token embedding，query embedding 复用 target 的
`text/token_embedding`。若该表中 id 248070 那一行的范数相对正常词表行是退化值
（≈0 或量级异常），就说明这个 token 在 target 预训练里从未被使用 ⇒ 草稿的 mask 位
拿到的输入是"未训练向量" ⇒ 第 1..K 位 hidden 必然不可用 ⇒ 链在位置 1 断。
"""

import json
import math
import pathlib
import struct
import sys

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
NAMES = ("text/token_embedding", "text/lm_head", "text/output_head")


def load_doc(path):
    with open(path, "rb") as handle:
        handle.read(8)
        (length,) = struct.unpack("<Q", handle.read(8))
        payload = 16 + length
        payload = (payload + 4095) // 4096 * 4096
        doc = json.loads(handle.read(length).decode("utf-8"))
    entries = {e["name"]: e for e in doc["objects"] if e.get("kind") == "tensor"}
    return entries, payload


entries, payload_start = load_doc(ART)
for name in NAMES:
    e = entries.get(name)
    print(f"{name}: {'MISSING' if e is None else (e.get('shape'), e.get('format'), e.get('bytes'))}")
print()

e = entries.get("text/token_embedding")
if e is None:
    print("没有 text/token_embedding，无法检验")
    raise SystemExit(1)

rows, cols = e["shape"]
fmt = e["format"]
print(f"embedding: {rows} x {cols}, format={fmt}")

# 只支持 BF16 直接读；其它格式先报出来
if fmt != "BF16":
    print(f"（{fmt} 需要反量化，本轮先只报格式）")
    raise SystemExit(0)

row_bytes = cols * 2
mask_id = 248070
probe_ids = [mask_id - 2, mask_id - 1, mask_id, mask_id + 1, mask_id + 2,
             0, 1, 100, 1000, 10000, 100000, 200000, 248044]


def row_norm(handle, idx):
    handle.seek(payload_start + e["offset"] + idx * row_bytes)
    raw = handle.read(row_bytes)
    values = struct.unpack(f"<{cols}e", raw)
    return math.sqrt(sum(float(v) * float(v) for v in values)), values


with open(ART, "rb") as handle:
    norms = {}
    for idx in probe_ids:
        if 0 <= idx < rows:
            norms[idx], _ = row_norm(handle, idx)

    # 采样若干普通行做参照
    sample = [7, 17, 370, 3709, 8888, 45678, 123456, 222222, 240000]
    ref = {}
    for idx in sample:
        if idx < rows:
            ref[idx], _ = row_norm(handle, idx)

print("普通采样行的 L2 范数：")
for idx in sorted(ref):
    print(f"   id={idx:<8} |row| = {ref[idx]:.4f}")
avg = sum(ref.values()) / max(1, len(ref))
print(f"   采样均值 = {avg:.4f}")
print()
print("mask 附近行：")
for idx in sorted(norms):
    tag = "  <-- MASK TOKEN" if idx == mask_id else ""
    ratio = (norms[idx] / avg) if avg > 0 else float("nan")
    print(f"   id={idx:<8} |row| = {norms[idx]:.4f}   (相对采样均值 {ratio:.3f}x){tag}")
print()

m = norms.get(mask_id)
if m is None:
    print("mask 行读取失败")
else:
    if m < 0.05:
        print("⇒ mask 行范数 ≈ 0：**该 embedding 从未被训练**（退化行）")
    elif m < 0.3 * avg:
        print(f"⇒ mask 行范数显著低于正常行（{m/avg:.3f}x）：很可能是未训练/退化")
    elif m > 3.0 * avg:
        print(f"⇒ mask 行范数是异常离群值（{m/avg:.3f}x）：需人工看一眼")
    else:
        print(f"⇒ mask 行范数与正常行同量级（{m/avg:.3f}x）：不能由范数判定未训练，"
              f"需换判据（例如与邻居行的夹角/是否等于 embedding 初始化分布）")
