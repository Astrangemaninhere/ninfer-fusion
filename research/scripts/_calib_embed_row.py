#!/usr/bin/env python3
"""定标 FP8_E4M3FN_ROW_BF16S / row-scale-v1 的行内布局，然后判 mask 行。

判据：E4M3 的 |v| 上限是 448 ⇒ 反量化后的 max|v| 必须 ≤ 448 才可信；
正常 embedding 行的 L2 范数应在 1~50 量级。
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
rows, cols = entry["shape"]
stride = cols + 2


def e4m3(b):
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0x0F
    m = b & 0x07
    if e == 0:
        return 0.0 if m == 0 else s * (m / 8.0) * 2.0 ** -6
    if e == 15 and m == 7:
        return float("nan")
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def read_row(handle, idx):
    handle.seek(payload + entry["offset"] + idx * stride)
    return handle.read(stride)


def as_float16(raw):
    return struct.unpack_from("<e", raw, 0)[0]


with open(ART, "rb") as handle:
    raw = read_row(handle, 220)
    print("行内前 8 字节:", raw[:8].hex(), " 后 8 字节:", raw[-8:].hex())
    print("把行首 2 字节当 bf16:", as_float16(raw[0:2]),
          " 把行尾 2 字节当 bf16:", as_float16(raw[cols:cols + 2]))
    print()
    for label, vals_off, scale_off in (("scale 在行尾", 0, cols),
                                       ("scale 在行首", 2, 0)):
        scale = as_float16(raw[scale_off:scale_off + 2])
        vals = [e4m3(b) * scale for b in raw[vals_off:vals_off + cols]]
        mx = max(abs(v) for v in vals)
        norm = math.sqrt(sum(v * v for v in vals))
        print(f"{label}: scale={scale:.6f}  max|v|={mx:9.3f}  |row|={norm:9.3f}"
              f"   {'可信' if mx <= 448.5 else '不可信(>448)'}")
    print()
    print("=== 用可信布局统计若干行 ===")
    for label, vals_off, scale_off in (("scale 在行尾", 0, cols),
                                       ("scale 在行首", 2, 0)):
        scale = as_float16(raw[scale_off:scale_off + 2])
        vals = [e4m3(b) * scale for b in raw[vals_off:vals_off + cols]]
        mx = max(abs(v) for v in vals)
        if mx > 448.5:
            continue
        print(f"采用布局：{label}")
        for idx in (0, 1, 15, 220, 3709, 100000, 200000, 247000,
                    248044, 248046, 248068, 248069, 248070, 248071, 248072):
            r = read_row(handle, idx)
            sc = as_float16(r[scale_off:scale_off + 2])
            vs = [e4m3(b) * sc for b in r[vals_off:vals_off + cols]]
            n = math.sqrt(sum(v * v for v in vs))
            m = max(abs(v) for v in vs)
            tag = "  <-- MASK TOKEN" if idx == 248070 else ""
            print(f"   id={idx:<8} |row|={n:8.3f}  max|v|={m:8.3f}  scale={sc:9.5f}{tag}")
        print()
        break
