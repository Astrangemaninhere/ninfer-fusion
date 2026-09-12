#!/bin/bash
# 补头文件 + touch 运行时 TU + 编译
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
LOG=$J/dl/build_featdump.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 编译（特征链仪表） $(date '+%F %H:%M:%S') ==="

python3 - <<'PY'
import pathlib
F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
t = F.read_text()
if "#include <cstdio>" not in t:
    lines = t.splitlines()
    # 插到第一个 #include 之后（若无则文件头）
    idx = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#include"):
            idx = i + 1
    lines[idx:idx] = ["#include <cstdio>", "#include <cstdlib>"]
    F.write_text("\n".join(lines) + "\n")
    print("已补 <cstdio>/<cstdlib>")
else:
    print("已存在")
PY

echo "--- touch 运行时 TU ---"
n=0
for h in dflash2_impl.h dflash_impl.h instantiate.h; do
  for inc in $(grep -rl "$h" $R/src 2>/dev/null | grep -v '\.orig' | sed "s|$R/||"); do
    case "$inc" in *.cpp|*.cu) touch "$R/$inc"; n=$((n+1));; esac
  done
done
echo "  touched $n"

cd "$R/build" || exit 4
start=$(date +%s)
make ninfer -j3 2>&1 | tail -6
rc=${PIPESTATUS[0]}
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 5; }
ls -l --time-style=+%H:%M $R/build/apps/ninfer | awk '{print "  bin:", $5, $6}'
echo BUILD_FEATDUMP_OK
