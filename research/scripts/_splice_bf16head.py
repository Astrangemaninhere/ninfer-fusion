#!/usr/bin/env python3
"""把 artifact 的 text/output_head 由 FP8 行缩放重编码为 BF16（单变量实验）。

只改一个对象：格式 FP8_E4M3FN_ROW_BF16S -> BF16，layout -> contiguous-le-v1。
identity 与其它对象逐字节不变（引擎只做结构校验）。
"""
import shutil
import sys

import torch

sys.path.insert(0, "/home/user/ninfer-fusion")

from tools.artifact.container import (  # noqa: E402
    Artifact,
    ResourceSpec,
    TensorSpec,
    write_artifact,
)
from tools.artifact.layouts import (  # noqa: E402
    decode_fp8_row_scaled_words,
    encode_direct,
)

OLD = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
NEW = "/home/user/models/qwen3_8_27b_nvfp4_dflash2_bf16head.ninfer"
TARGET = "text/output_head"
BLK = 8192

src = Artifact(OLD)
obj = src.find(TARGET)
print(f"目标对象: {obj.name} shape={tuple(obj.shape)} format={obj.format} "
      f"layout={obj.layout} bytes={obj.bytes}")

# 1) 分块解码 FP8 -> BF16 字节
codes, scales = decode_fp8_row_scaled_words(src.payload(obj), obj.shape)
n, k = tuple(obj.shape)
assert tuple(codes.shape) == (n, k), (codes.shape, (n, k))
print(f"  FP8 codes {tuple(codes.shape)} scales {tuple(scales.shape)}")

parts = []
for off in range(0, n, BLK):
    end = min(off + BLK, n)
    blk = codes[off:end].view(torch.float8_e4m3fn).float() * scales[off:end].float().unsqueeze(1)
    parts.append(encode_direct(blk.to(torch.bfloat16), "BF16"))
new_payload = b"".join(parts)
del codes, scales, parts
expect = n * k * 2
assert len(new_payload) == expect, (len(new_payload), expect)
print(f"  BF16 payload = {len(new_payload)} 字节（{len(new_payload)/2**30:.2f} GiB）")

# 2) 逐对象重写（mmap 零拷贝 + 单对象替换）
entries = []
count = 0
for o in src.objects:
    if o.name == TARGET:
        entries.append((TensorSpec(o.name, tuple(o.shape), "BF16", "contiguous-le-v1"),
                        new_payload))
    elif o.kind == "tensor":
        entries.append((TensorSpec(o.name, tuple(o.shape), o.format, o.layout),
                        src.payload(o)))
    else:
        entries.append((ResourceSpec(o.name, o.encoding, o.bytes), src.payload(o)))
    count += 1
print(f"  待写对象 {count} 个；identity=({src.identity.model_id}, {src.identity.weights_id})")

write_artifact(NEW, src.identity, entries)
src.close()
print("写出完成 ->", NEW)
