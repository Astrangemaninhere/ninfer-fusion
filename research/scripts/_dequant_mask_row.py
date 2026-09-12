#!/usr/bin/env python3
"""反量化 token_embedding 的 mask 行，检验它是否"从未被训练"。

布局：FP8_E4M3FN_ROW_BF16S = 每行 [5120 个 E4M3 字节][2 字节 BF16 scale]。
"""

import json
import math
import struct

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
NAME = "text/token_embedding"

with open(ART, "rb") as handle:
    handle.read(8)
    (length,) = struct.unpack("<Q", handle.read(8))
    payload = 16 + length
    payload = (payload + 4095) // 4096 * 4096
    doc = json.loads(handle.read(length).decode("utf-8"))

entry = next(e for e in doc["objects"] if e.get("name") == NAME)
print("entry:", json.dumps({k: v for k, v in entry.items() if k != "name"}))
rows, cols = entry["shape"]
stride = cols + 2                                   # 每行 FP8 值 + BF16 scale
assert entry["bytes"] == rows * stride, (entry["bytes"], rows * stride)
print(f"rows={rows} cols={cols} stride={stride} 一致 ✓")
print()


def e4m3_to_float(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if exp == 0:
        if mant == 0:
            return 0.0
        return sign * (mant / 8.0) * (2.0 ** -6)
    if exp == 15 and mant == 7:
        return float("nan")
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def row_stats(handle, idx):
    handle.seek(payload + entry["offset"] + idx * stride)
    raw = handle.read(stride)
    scale = struct.unpack_from("<e", raw, cols)[0]
    vals = [e4m3_to_float(b) * scale for b in raw[:cols]]
    norm = math.sqrt(sum(v * v for v in vals))
    mx = max(abs(v) for v in vals)
    mean = sum(vals) / len(vals)
    return norm, mx, mean, scale


probe = [248068, 248069, 248070, 248071, 248072, 248044, 248046,
         0, 1, 15, 220, 3709, 100000, 200000, 247000]
with open(ART, "rb") as handle:
    stats = {i: row_stats(handle, i) for i in probe if i < rows}

ref_norms = [stats[i][0] for i in (0, 1, 15, 220, 3709, 100000, 200000, 247000)]
avg = sum(ref_norms) / len(ref_norms)
print("普通行（作参照）：")
for i in (0, 1, 15, 220, 3709, 100000, 200000, 247000):
    if i in stats:
        n, mx, mean, sc = stats[i]
        print(f"   id={i:<8} |row|={n:8.3f}  max|v|={mx:7.3f}  mean={mean:8.4f}  scale={sc:.5f}")
print(f"   ⇒ 参照均值 |row| = {avg:.3f}")
print()
print("mask 附近与特殊 token：")
for i in (248044, 248046, 248068, 248069, 248070, 248071, 248072):
    if i not in stats:
        continue
    n, mx, mean, sc = stats[i]
    tag = "  <-- MASK TOKEN (草稿块第1..K位的输入)" if i == 248070 else ""
    print(f"   id={i:<8} |row|={n:8.3f} ({n/avg:5.3f}x)  max|v|={mx:7.3f}  "
          f"mean={mean:8.4f}  scale={sc:.5f}{tag}")
print()
m = stats[248070][0]
ratio = m / avg
if m < 1e-6:
    print("⇒ mask 行全 0：**该 embedding 从未被训练**")
elif ratio < 0.3:
    print(f"⇒ mask 行范数仅 {ratio:.3f}x 正常值：强烈提示未训练/退化")
elif ratio > 3.0:
    print(f"⇒ mask 行范数 {ratio:.3f}x 正常值：异常离群，需人看")
else:
    print(f"⇒ mask 行范数 {ratio:.3f}x 正常值：范数不能判定；改看该行与其它行的相似度")
