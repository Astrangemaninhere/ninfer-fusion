#!/bin/bash
# 定案：verify(W>=4) 现在到底走不走 A4（激活量化 FP4）路线？
#   1) 读源码常量（A 路说是从 variant.cpp 推断的）
#   2) 读 artifact 里 GDN 相关权重的格式（有没有 FP4 档）
R=/home/user/ninfer-fusion
echo "=== 1) variant.cpp:55-75（AllowA4 / A16Only 常量） ==="
sed -n '55,75p' "$R/src/targets/qwen3_6_27b/impl/variant.cpp"
echo
echo "=== 2) 该常量/枚举的定义与使用点 ==="
grep -rn "AllowA4\|A16Only" "$R/src" 2>/dev/null | sed "s|$R/||" | head -12
echo
echo "=== 3) artifact 里 GDN 权重格式分布（看有没有 FP4 档） ==="
python3 - <<'PY'
import json, struct, collections
ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART, "rb") as h:
    h.read(8); (L,) = struct.unpack("<Q", h.read(8)); doc = json.loads(h.read(L).decode())
cnt = collections.Counter()
gdn = collections.Counter()
for e in doc["objects"]:
    if e.get("kind") != "tensor":
        continue
    cnt[e.get("format")] += 1
    if "/gdn/" in e["name"]:
        gdn[(e["name"].split("/gdn/")[-1].split("/")[0], e.get("format"))] += 1
print("全 artifact 格式分布:", dict(cnt))
print("GDN 子项格式分布:")
for k, v in sorted(gdn.items()):
    print(f"   {k[0]:<28} {k[1]:<22} x{v}")
PY
