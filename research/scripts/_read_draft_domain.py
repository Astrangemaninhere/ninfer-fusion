import json, struct, pathlib, re, collections, numpy as np

ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")
with open(ART, "rb") as f:
    f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}

o = objs.get("text/draft_head_token_ids")
print("draft_head_token_ids:", {k: o.get(k) for k in ("kind", "format", "shape", "bytes")} if o else "缺失")
if o:
    with open(ART, "rb") as f:
        f.seek(payload + o["offset"])
        ids = np.frombuffer(f.read(o["bytes"]), dtype=np.int32)
    print("  条目 =", ids.shape, " 范围 = [%d, %d]" % (ids.min(), ids.max()))
    print("  前 12 个 =", ids[:12].tolist())
    print("  唯一值数 =", len(np.unique(ids)), " (域大小 131072?)")
    # 看看是不是排序过的子集
    print("  是否排序 =", bool(np.all(np.diff(ids) > 0)))
    np.save("/home/user/draft_domain_ids.npy", ids)

# 明确一下 head 的输出域：用 head 的行数 = 131072；若 token_ids 是其映射表，则候选 -> 真词表 id
print()
print("=== artifact 里与 proposal/draft 有关的对象一览 ===")
for n in sorted(objs):
    if any(k in n for k in ("draft_head", "output_head", "mtp/", "token_embedding")):
        z = objs[n]
        print("  %-34s %-18s %-16s bytes=%s" % (n, z.get("format"), str(z.get("shape")), z.get("bytes")))
