#!/bin/bash
# MTP 路径的提案来源取证（定点 grep，只读）
R=/home/user/ninfer-fusion
echo "=== 1) mtp_impl / dflash_impl 里用了哪些头 ==="
grep -rnE 'mtp|output_head|lm_head|draft_head|embed' $R/src/targets/qwen3_6/impl/runtime/mtp_impl.h 2>/dev/null | head -25
echo
echo "=== 2) mtp 提案函数的签名与关键行 ==="
grep -rnE 'void .*propose|Tensor& logits|argmax|topk|top_k|candidates' $R/src/targets/qwen3_6/impl/runtime/mtp_impl.h $R/src/targets/qwen3_6/impl/runtime/dflash_impl.h 2>/dev/null | head -25
echo
echo "=== 3) artifact 里 mtp/* 与 dflash2/* 的对象数与格式对照 ==="
grep -n 'mtp' /mnt/c/Users/User/Documents/ziqinzhang/_art_quant_names.py >/dev/null 2>&1
python3 - <<'PY'
import json, struct, pathlib, collections
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")
with open(ART, "rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
g = collections.defaultdict(lambda: [0, 0])
for o in j["objects"]:
    if o.get("kind") != "tensor": continue
    ns = o["name"].split("/")[0]
    if ns in ("mtp", "dflash2", "text", "vision"):
        g[ns][0] += 1; g[ns][1] += o.get("bytes", 0)
for ns, (n, b) in sorted(g.items()):
    print("  %-10s 张量 %4d  %.3f GiB" % (ns, n, b/2**30))
print()
for o in j["objects"]:
    if o.get("kind") == "tensor" and o["name"].startswith("mtp/"):
        print("  %-44s %-16s %-18s %d B" % (o["name"], o.get("format"), str(o.get("shape")), o.get("bytes", 0)))
PY
