#!/usr/bin/env python3
import json
import numpy as np

NF = ("/mnt/c/Users/User/Documents/ziqinzhang/models/"
      "Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4/qwen3_8_27b_nvfp4.ninfer")
f = open(NF, "rb")
f.read(8)
n = int.from_bytes(f.read(8), "little")
hdr = json.loads(f.read(n).decode())
meta_end = 16 + n
print("metadata_end =", meta_end)
for A in (512, 1024, 2048, 4096, 8192, 16384):
    po = ((meta_end + A - 1) // A) * A
    f.seek(po)
    b = f.read(20)
    print("  align %-6d payload_offset=%-8d first20=%r" % (A, po, b))

o = [x for x in hdr["objects"] if x["name"] == "text/token_embedding"][0]
for po in sorted({((meta_end + A - 1) // A) * A for A in (4096, 8192)}):
    f.seek(po + o["offset"])
    b = np.frombuffer(f.read(32), dtype=np.uint8)
    print("payload_offset=%d token_embedding.first32=%s" % (po, b.tolist()))
f.close()
