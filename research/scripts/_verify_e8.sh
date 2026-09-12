#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== [verify E8 claim A] our shared helper ==="
grep -n -A6 'inline .*silu' "$R/src/ops/common/math.cuh" | head -24 | cut -c1-120
echo
echo "=== [verify E8 claim B] call-site count (should be ~66 in ~18 files) ==="
grep -rlE '\bsilu\s*\(' "$R/src" --include=*.cu --include=*.cuh 2>/dev/null | wc -l | sed 's/^/  files: /'
grep -rhoE '\bsilu\s*\(' "$R/src" --include=*.cu --include=*.cuh 2>/dev/null | wc -l | sed 's/^/  call sites: /'
echo
echo "=== E8 diff target ==="
head -12 "$J/_collab/E8_s51_nvfp4_silu.diff"
echo
echo "=== queue for the NEXT build (this build must stay Patch-A-only in attribution) ==="
python3 - <<'PY'
import datetime, pathlib
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 10. 下一趟编译队列（刻意不塞进本轮，避免污染补丁 A 的归因，%s）
1. **S51 / E8**：`ops::silu` 的近似式在 x 很负时被归零（我们树是"精确 expf + IEEE 除法"，零点在
   x = -88.72284；此时真值 SiLU = -2.607e-37，**是 bf16 最小正规数的 22.18 倍**、最小次正规数的 2839 倍
   ⇒ 不是次正规噪声，是真的精度损失）。上游 PR #194 把指数折到不会溢出的一侧（除数恒在 (1,2]）。
   我们树把上游的 file-local `swiglu_silu` 合并成了共享 `ops::silu`（`src/ops/common/math.cuh`），
   所以 **一处修复覆盖全部调用点**（E8 扫到 66 处 / 18 文件）。diff 已就绪（+18/-3），
   与 mirror 的 dry-run 均 rc=0（严格 --fuzz=0）。**推迟理由**：它改的是所有 nvfp4/bf16 线性层的数值，
   会让补丁 A 的接受率归因变浑。
2. **E7 的回归测试** `E7_s50_regression_sketch.diff`（+36 行 store 级页边界用例）：需要能编 tests 的窗口。
3. **E9 的 dflash2 可配置 K 最小切片**（若它给出 ≤60 行 diff）：同样会改 dflash2 行为，单独一轮。
4. E3 的 i8 平面步长 + S36 恢复（256 下逐字节等价）、E1 的补丁 B（attention_valid 契约）。
""" % datetime.datetime.now().strftime("%H:%M"))
print("  queued")
PY
echo
echo "=== build progress ==="
grep -oE '^\[[ 0-9]+%\][^"]*' /tmp/pa_make_1.log | tail -1
for p in $(pgrep -f bin/nvcc); do echo "  nvcc elapsed $(ps -o etime= -p $p | tr -d ' ')"; break; done
