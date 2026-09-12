#!/usr/bin/env python3
"""把 bf16head artifact 的 weights_id 改成 nvfp4-dflash2-bf16head（就地改，payload_start 不变）。"""
import json
import pathlib
import struct

P = pathlib.Path("/home/user/models/qwen3_8_27b_nvfp4_dflash2_bf16head.ninfer")
with P.open("rb") as f:
    magic = f.read(8)
    (n,) = struct.unpack("<Q", f.read(8))
    doc = json.loads(f.read(n).decode())
print("magic =", magic, " 原 json_len =", n)
print("原 identity =", doc["identity"])
assert doc["identity"]["weights_id"] == "nvfp4-dflash2"

doc["identity"]["weights_id"] = "nvfp4-dflash2-bf16head"
new = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
limit = 192512 - 16
if len(new) > limit:
    raise SystemExit(f"新 JSON {len(new)} 超过 {limit}，payload_start 会移动；改用整体重写")
print(f"新 json_len = {len(new)}（上限 {limit}）；payload_start 保持 {192512}")

with P.open("r+b") as f:
    f.write(magic)
    f.write(struct.pack("<Q", len(new)))
    f.write(new)
    f.write(b"\0" * (192512 - 16 - len(new)))  # 复位原尾部对齐填充
print("就地改写完成")
