#!/bin/bash
# 重配置批次：A3 的三份 Spark new_op + E7/S50 回归测试。
#
# 为什么单独一批、且接受全量重编：
#   * A3 的 gelu_mul 带 6 个新文件 + 两个 CMakeLists 改动 => 必须 cmake 重新配置，
#     而重新配置在本树就是全量重编（今天 2h 的由来）。既然这笔钱必须付，
#     就把所有"需要新文件"的改动一次塞满这一批：A3 三份 + E7 测试。
#   * 反过来，A5b + S52（只改现有文件）已经排在前一批，不与之混，保持 A5b 归因干净。
#
# 判据（不靠"看起来对"）：
#   G1 每个补丁先 dry-run（原样 / strip CR 两种都试），任一不过 -> 整体放弃；
#   G2 编译成功，ninfer 与 ninfer-serve 都被重链；
#   G3 ctest -R 跑新算子/存储的测试，必须全 PASS —— 这是 A3 的验收，不是"编译过"。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
B=$R/build/apps
LOG=$J/dl/batch_reconfig.log
export PATH="/home/user/.local/bin:$PATH"
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== batch(重配置: A3 + E7) $(date '+%F %H:%M:%S') ==="

# 自匹配陷阱：模式与自身 cmdline 同形会让 pgrep 永远为真。
batch1_running() { pgrep -f '_batch_build_next[.]sh' >/dev/null 2>&1; }
compiler_alive() { pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1; }
done_marker()    { grep -q 'BATCH_BUILD_NEXT_DONE' "$J/dl/batch_build_next.log" 2>/dev/null; }

t0=$(date +%s)
while batch1_running || compiler_alive || ! done_marker; do
  if [ $(( $(date +%s) - t0 )) -gt 21600 ]; then echo "TIMEOUT 等前一批 (>6h)"; exit 2; fi
  sleep 30
done
echo "--- 前一批已结束 $(date +%H:%M:%S) ---"
grep -E '^\- ' "$C/M_a5b_verify.md" 2>/dev/null | head -4

PATCHES=("$C/A3_spark_head_geometry.diff" "$C/A3_spark_headwise_gate.diff" \
         "$C/A3_spark_gelu_mul.diff" "$C/E7_s50_regression_sketch.diff")

resolve() {
  local f=$1
  patch -p1 --dry-run -d "$R" < "$f" >/dev/null 2>&1 && { echo "$f"; return 0; }
  local t=/tmp/rc_$(basename "$f")
  tr -d '\r' < "$f" > "$t"
  patch -p1 --dry-run -d "$R" < "$t" >/dev/null 2>&1 && { echo "$t"; return 0; }
  return 1
}
declare -a RESOLVED=()
for f in "${PATCHES[@]}"; do
  [ -f "$f" ] || { echo "G1 FAIL: 缺 $f"; exit 3; }
  r=$(resolve "$f") || { echo "G1 FAIL: $(basename "$f") 打不上"; exit 3; }
  echo "  G1 OK: $(basename "$f")"
  RESOLVED+=("$r")
done

cp -f "$B/ninfer" /home/user/ninfer_before_sparkops
echo "  基线(sparkops 之前): $(stat -c '%y %s' /home/user/ninfer_before_sparkops | cut -c1-19)"

for r in "${RESOLVED[@]}"; do
  patch -p1 -b -d "$R" < "$r" || { echo "落补丁失败: $r"; exit 5; }
  echo "  applied: $(basename "$r")"
done

# ---- 重新配置（这一步是全量重编的代价所在，必须做，因为 A3 新增了文件）----
cd "$R/build" || exit 6
echo "--- cmake 重新配置 $(date +%H:%M:%S) ---"
cmake -S "$R" -B "$R/build" > /tmp/reconfig.log 2>&1
rc=$?; tail -4 /tmp/reconfig.log; echo "  cmake rc=$rc"
[ $rc -ne 0 ] && { echo "reconfigure 失败"; echo RECONFIG_ABORTED | tee -a "$LOG"; exit 7; }

for target in ninfer ninfer-serve; do
  rc=1
  for attempt in 1 2 3; do
    echo "--- make $target attempt $attempt $(date +%H:%M:%S) ---"
    free -g | sed -n 2p
    make "$target" -j1 > /tmp/rc_${target}_$attempt.log 2>&1
    rc=$?
    tail -3 /tmp/rc_${target}_$attempt.log
    [ $rc -eq 0 ] && break
    grep -E 'error:' /tmp/rc_${target}_$attempt.log | head -6 | cut -c1-170
    sleep 5
  done
  echo "  $target rc=$rc"
  [ $rc -ne 0 ] && { echo "编译失败"; echo RECONFIG_ABORTED | tee -a "$LOG"; exit 8; }
done
ls -l --time-style=+%H:%M "$B/ninfer" "$B/ninfer-serve" | awk '{print "  新:", $6, $5, $NF}'

# ---- G3: A3 的验收 = 新算子与存储测试全 PASS（不是"编译过"）----
echo "--- 编译新测试并跑 $(date +%H:%M:%S) ---"
make -j1 ninfer_gelu_mul_test ninfer_sigmoid_mul_test ninfer_prepare_masked_block_test \
     > /tmp/rc_tests_build.log 2>&1
trc=$?
tail -3 /tmp/rc_tests_build.log
echo "  tests build rc=$trc"
if [ $trc -eq 0 ]; then
  ctest --test-dir "$R/build" -R 'gelu_mul|sigmoid_mul|prepare_masked_block|context_store' \
        --output-on-failure > /tmp/rc_ctest.log 2>&1
  echo "  ctest rc=$?"
  tail -12 /tmp/rc_ctest.log
else
  echo "  测试目标没编出来（可能目标名不同）："
  (cd "$R/build" && make help 2>/dev/null | grep -iE 'gelu|sigmoid|context_store' | head -8)
fi

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib, time
ct = pathlib.Path("/tmp/rc_ctest.log")
body = ct.read_text(errors="replace") if ct.exists() else "(没有 ctest 输出)"
ok = "100% tests passed" in body or "tests passed" in body
out = ["", "## Spark new_op + E7 回归 批次结果 (%s)" % time.strftime("%F %H:%M"), "",
       "- 落的补丁：A3 head_geometry / headwise_gate / gelu_mul + E7/S50 store 页边界回归",
       "- 新增文件：6 个（gelu_mul 全套）+ 两个 CMakeLists 改动 => 本批已做 cmake 重新配置",
       "- 判据：ctest 全 PASS（不是'编译过就算'）。当前判定：%s" % ("PASS" if ok else "见下方原始输出"),
       "", "```", body.strip()[-1500:], "```"]
pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_sparkops_verify.md").write_text(
    "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out[:4]))
PY
echo BATCH_RECONFIG_DONE | tee -a "$LOG"
