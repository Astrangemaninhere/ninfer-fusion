import json, struct

ART = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART, "rb") as f:
    f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
objs = j["objects"]

print("=== 非 BF16/FP32 的量化张量：按 format 归类，看名字模式 ===")
from collections import defaultdict
g = defaultdict(list)
for o in objs:
    if o.get("kind") != "tensor":
        continue
    fmt = o.get("format")
    if fmt in ("BF16", "FP32", "I32"):
        continue
    g[fmt].append(o)
for fmt, lst in sorted(g.items()):
    print("--- %s  共 %d 个" % (fmt, len(lst)))
    for o in lst[:6]:
        print("      %-58s %s" % (o["name"], str(o.get("shape"))))
    if len(lst) > 6:
        print("      ...")
    # 名字里出现的命名空间前缀
    pref = defaultdict(int)
    for o in lst:
        parts = o["name"].split("/")
        pref["/".join(parts[:2])] += 1
    print("      前缀分布:", dict(sorted(pref.items(), key=lambda x: -x[1])[:6]))
