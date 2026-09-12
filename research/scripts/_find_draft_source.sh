#!/bin/bash
echo "=== 1) artifact 里非 tensor 的条目（可能带出处/配置） ==="
python3 - <<'PY'
import json, struct
ART="/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART,"rb") as h:
    h.read(8); (L,)=struct.unpack("<Q",h.read(8)); doc=json.loads(h.read(L).decode())
print("identity:", doc.get("identity"))
print("doc keys:", sorted(doc.keys()))
for it in doc["objects"]:
    if it.get("kind") != "tensor":
        print(f"  {it.get('name'):<44} {it.get('kind')} {it.get('encoding')} {it.get('bytes')}")
PY
echo
echo "=== 2) /home/user 下的草稿来源痕迹 ==="
ls -d /home/user/*dflash* /home/user/*draft* /home/user/models/* 2>/dev/null | head -20
echo
echo "=== 3) 项目里构建 dflash2 artifact 的脚本/记录 ==="
grep -rln "dflash2" /mnt/c/Users/User/Documents/ziqinzhang/*.py /mnt/c/Users/User/Documents/ziqinzhang/*.sh /mnt/c/Users/User/Documents/ziqinzhang/*.bat 2>/dev/null | head -12
echo "--- 提到 dflash2 且提到下载/转换的行 ---"
grep -rn "dflash2" /mnt/c/Users/User/Documents/ziqinzhang/_collab/*.md 2>/dev/null | grep -iE "下载|download|转换|convert|来源|hf|hugging|artifact 构建" | head -14
echo
echo "=== 4) dspark artifact 的同名出处 ==="
python3 - <<'PY'
import json, struct
for a in ("qwen3_8_27b_nvfp4_dspark.ninfer",):
    ART=f"/home/user/models/{a}"
    with open(ART,"rb") as h:
        h.read(8); (L,)=struct.unpack("<Q",h.read(8)); doc=json.loads(h.read(L).decode())
    print(a, "identity:", doc.get("identity"))
    names=[it["name"] for it in doc["objects"] if it.get("kind")=="tensor"]
    print("   groups:", sorted({n.split('/')[0] for n in names}))
PY
