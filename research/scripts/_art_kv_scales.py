import json, struct, pathlib

ART = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART, "rb") as f:
    f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
objs = j["objects"]

# identity 里可能就写着 KV 布局
print("=== identity ===")
print(json.dumps(j.get("identity", {}), ensure_ascii=False)[:1200])
print()
print("=== 含 scale / kv / cache 字样的对象 ===")
hits = [o for o in objs if any(k in o["name"].lower() for k in ("scale", "kv", "cache"))]
print("  count =", len(hits))
for o in hits[:40]:
    print("  %-46s kind=%-9s format=%-12s shape=%-18s bytes=%s" %
          (o["name"], o.get("kind"), o.get("format"), str(o.get("shape")), o.get("bytes")))
print()
print("=== 按 kind 统计 ===")
from collections import Counter
print(" ", Counter(o.get("kind") for o in objs))
print("=== tensor 的 format 分布 ===")
print(" ", Counter(o.get("format") for o in objs if o.get("kind") == "tensor"))
