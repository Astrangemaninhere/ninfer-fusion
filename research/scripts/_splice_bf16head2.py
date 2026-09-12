#!/usr/bin/env python3
"""v2：把 text/output_head 与 text/token_embedding 一并由 FP8 重编码为 BF16，
identity 直接写成新档 nvfp4-dflash2-bf16head。

理由：bindings.cpp 的 endpoint_format 同时决定这两个张量的精度；
上游 vLLM #52816 要求 DFlash2 的候选 TopK 必须来自"未量化的目标 LM 头"。
"""
import sys

import torch

sys.path.insert(0, "/home/user/ninfer-fusion")

from tools.artifact.container import (  # noqa: E402
    Artifact,
    ArtifactIdentity,
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
TARGETS = ("text/output_head", "text/token_embedding")
BLK = 8192

src = Artifact(OLD)
payloads: dict[str, bytes] = {}
for name in TARGETS:
    obj = src.find(name)
    codes, scales = decode_fp8_row_scaled_words(src.payload(obj), obj.shape)
    n, k = tuple(obj.shape)
    assert tuple(codes.shape) == (n, k)
    parts = []
    for off in range(0, n, BLK):
        end = min(off + BLK, n)
        blk = codes[off:end].view(torch.float8_e4m3fn).float() * scales[off:end].float().unsqueeze(1)
        parts.append(encode_direct(blk.to(torch.bfloat16), "BF16"))
    payloads[name] = b"".join(parts)
    del codes, scales, parts
    assert len(payloads[name]) == n * k * 2
    print(f"{name}: {obj.format} {obj.bytes} -> BF16 {len(payloads[name])} 字节")

entries = []
for o in src.objects:
    if o.name in payloads:
        entries.append((TensorSpec(o.name, tuple(o.shape), "BF16", "contiguous-le-v1"),
                        payloads[o.name]))
    elif o.kind == "tensor":
        entries.append((TensorSpec(o.name, tuple(o.shape), o.format, o.layout), src.payload(o)))
    else:
        entries.append((ResourceSpec(o.name, o.encoding, o.bytes), src.payload(o)))

identity = ArtifactIdentity("qwen3.8-27b", "nvfp4-dflash2-bf16head")
print("identity ->", identity)
write_artifact(NEW, identity, entries)
print("写出完成 ->", NEW)
sys.stdout.flush()
# 不用 close()：entries 仍持有 mmap 导出的 memoryview（会抛 BufferError，属无害）
