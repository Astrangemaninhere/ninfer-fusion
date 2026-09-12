#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$R" || exit 3
echo "=== [1] 复核 S53 自检（我自己跑，不依赖它的日志） ==="
PYTHONPATH="$R" timeout 900 python3 tools/convert/check_source_map.py --full-materialize 2>&1 | tail -6
echo
echo "=== [2] 复核改判后的产物：BLOCKED 是否让位给 config.h ==="
ls -l --time-style=+%H:%M "$R/tools/archkit/out/spark-x2.5-4b/" | awk '{print "  ", $6, $5, $NF}'
echo
echo "=== [3] manifest 里那条 tied 行现在是什么 tier ==="
python3 - <<'PY'
import json, pathlib
m = json.loads(pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/out/spark-x2.5-4b/manifest.json").read_text())
for g in m.get("gaps", []):
    if "tied" in g["need"]:
        print("  need :", g["need"])
        print("  tier :", g["tier"])
        print("  act  :", g["action"][:160])
print("  new_op 数量:", sum(1 for g in m["gaps"] if g["tier"] == "new_op"))
print("  covered 行 :", [g["need"] for g in m["gaps"] if g["tier"] == "covered"])
print("  embedding 段:", json.dumps(m.get("embedding") or m.get("source_binding") or {}, ensure_ascii=False)[:200])
PY
echo
echo "=== [4] 旧备份是否在（回滚只需改一个词） ==="
ls -l "$J/_collab/s53_scratch/" 2>/dev/null | head -6 | awk '{print "  ", $5, $NF}'
echo
echo "=== [5] 编译 / 下载 ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -2
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
