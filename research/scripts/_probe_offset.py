#!/usr/bin/env python3
"""用"可验证内容"定标 .ninfer 的 offset 约定：tokenizer.json 是 JSON 文本，起点必为 '{'。"""
import json, struct

P = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
with open(P, "rb") as f:
    magic = f.read(8)
    n = struct.unpack("<Q", f.read(8))[0]
    hdr_raw = f.read(n)
man = json.loads(hdr_raw.decode("utf-8", "replace"))
res = [o for o in man["objects"] if o.get("kind") == "resource"]
print("magic=%s json_len=%d" % (magic.hex(), n))
print("资源对象:")
for o in res:
    print("   %-40s offset=%-12d bytes=%-10d encoding=%s" %
          (o["name"], o["offset"], o["bytes"], o.get("encoding")))

tok = [o for o in res if o["name"].endswith("tokenizer.json")][0]
size = __import__("os").path.getsize(P)
print("\n文件大小 = %d" % size)

cands = {
    "file_start(0)": 0,
    "16+json_len": 16 + n,
    "8+json_len": 8 + n,
    "json_len": n,
    "16": 16,
}
print("\n各候选起点处的头 24 字节（tokenizer.json 应以 '{' 开头）:")
for tag, base in cands.items():
    with open(P, "rb") as f:
        f.seek(base + tok["offset"])
        head = f.read(24)
    print("   %-16s @ %-10d  %r" % (tag, base + tok["offset"], head[:24]))

print("\n=== 各资源在各候选约定下是否落在合理区间 ===")
for tag, base in cands.items():
    ok = True
    for o in res:
        end = base + o["offset"] + o["bytes"]
        if base + o["offset"] < 0 or end > size:
            ok = False
    print("   %-16s 全部资源在文件内: %s" % (tag, "是" if ok else "否"))

print("\n=== 若 '16+json_len' 正确，则张量区起点应等于最大资源末端 ===")
base = 16 + n
end_max = max(base + o["offset"] + o["bytes"] for o in res)
tens = [o for o in man["objects"] if o.get("kind") == "tensor"]
first_t = min(tens, key=lambda o: o["offset"])
print("   资源末端(绝对) = %d" % end_max)
print("   第一个张量 %s: manifest offset=%d -> 绝对 %d" % (first_t["name"], first_t["offset"], base + first_t["offset"]))
print("   差 = %d 字节" % (base + first_t["offset"] - end_max))
