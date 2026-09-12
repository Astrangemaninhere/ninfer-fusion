#!/usr/bin/env python3
import json
import numpy as np

NF = ("/mnt/c/Users/User/Documents/ziqinzhang/models/"
      "Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4/qwen3_8_27b_nvfp4.ninfer")

with open(NF, "rb") as f:
    print("magic", f.read(8))
    n = int.from_bytes(f.read(8), "little")
    hdr = json.loads(f.read(n).decode("utf-8"))
PAYLOAD = 8 + 8 + n
print("payload_offset", PAYLOAD, "header_json_bytes", n)

o = [x for x in hdr["objects"] if x["name"] == "text/token_embedding"][0]
print("obj:", o)

nrows, k = o["shape"]
with open(NF, "rb") as f:
    f.seek(PAYLOAD + o["offset"])
    codes = np.frombuffer(f.read(256 * k), dtype=np.uint8).reshape(256, k)
    f.seek(PAYLOAD + o["offset"] + nrows * k)
    sc_raw = f.read(256 * 2)

print("codes: min %d max %d" % (codes.min(), codes.max()))
print("num 0x7F in codes:", int((codes == 0x7F).sum()),
      " num 0xFF:", int((codes == 0xFF).sum()))
print("first 16 code bytes:", codes[0, :16].tolist())

sc = np.frombuffer(sc_raw, dtype="<f2")
print("scales as BF16: first 8 =", sc[:8].tolist())
print("scales nan count:", int(np.isnan(sc).sum()),
      " inf:", int(np.isinf(sc).sum()),
      " zero:", int((sc == 0).sum()))
print("num 0x7F in scale bytes:", int((np.frombuffer(sc_raw, dtype=np.uint8) == 0x7F).sum()),
      " num 0xFF:", int((np.frombuffer(sc_raw, dtype=np.uint8) == 0xFF).sum()))
print("scale bytes first 16:", np.frombuffer(sc_raw, dtype=np.uint8)[:16].tolist())
