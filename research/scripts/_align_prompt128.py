#!/usr/bin/env python3
"""按 token 数把 prompt 对齐到 128 的整数倍（S_A 的可证伪预测所需）。

tokenizer 直接从 artifact 的 frontend/tokenizer.json 取（offset 0），不加载模型。
"""
import json
import pathlib
import struct
import sys

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
OUT = pathlib.Path("/tmp/prompt_align128.txt")
SEG = "杭州地处长江三角洲南翼，浙江省北部，北靠天目山，南临钱塘江，水网密布、湖荡众多。"

with open(ART, "rb") as handle:
    handle.read(8)
    (length,) = struct.unpack("<Q", handle.read(8))
    payload = 16 + length
    payload = (payload + 4095) // 4096 * 4096
    doc = json.loads(handle.read(length).decode("utf-8"))
entry = next(e for e in doc["objects"] if e["name"] == "frontend/tokenizer.json")
handle = open(ART, "rb")
handle.seek(payload + entry["offset"])
tok_bytes = handle.read(entry["bytes"])
handle.close()
pathlib.Path("/tmp/tok.json").write_bytes(tok_bytes)
print(f"tokenizer.json 取出 {len(tok_bytes)} B")

try:
    from tokenizers import Tokenizer
except Exception as exc:                                   # noqa: BLE001
    print("没有 tokenizers 库，退回本方法:", exc)
    sys.exit(2)

tok = Tokenizer.from_file("/tmp/tok.json")
base = tok.encode(SEG).ids
print(f"每段 token 数 = {len(base)}")

best = None
for reps in range(20, 120):
    ids = tok.encode(SEG * reps).ids
    n = len(ids)
    if n % 128 == 0 and n >= 1024:
        best = (reps, n)
        break
if best is None:
    print("没找到 128 倍数（扩大 reps 范围）")
    sys.exit(3)
reps, n = best
text = SEG * reps
OUT.write_text(text, encoding="utf-8")
print(f"选定 reps={reps}  prompt tokens={n}  (n%128={n % 128})  写出 {OUT}")
