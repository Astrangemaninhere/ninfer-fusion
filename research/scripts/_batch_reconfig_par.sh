#!/bin/bash
# 激进并行构建批次：A3 + E7（重配置）+ 用 -j8 把 16 个核吃满，允许峰值溢到 SSD swap。
#
# 为什么敢激进（全部来自实测，不是乐观假设）：
#   * 单 ptxas RSS 11.5 GB，但 MemAvailable 只有 8 GB 时也没有 OOM、swap 仅用 83 MB
#     ⇒ 单次 ptxas 的**边际内存**远小于 RSS（大头是可回收/共享页）。
#   * 机器 16 核 16 线程（Intel Ultra 7 270K Plus，无 SMT），构建当前只用 1 核 ⇒ 最大浪费在 CPU。
#   * swap 16 GB 在磁盘上，根盘余量 287 GB ⇒ 允许"溢到 SSD、慢一点"这条路线成立。
#   * 这批**不与模型同跑**（_par_build.sh 会拒绝启动），所以 OOM 的受害者只会是编译器（可重试）。
#
# 判据（不靠"看起来快了"）：
#   * 编译必须成功，ninfer / ninfer-serve 都被重链；
#   * ctest 跑新算子与存储测试全 PASS；
#   * 记下 -j8 的墙钟与 swap 峰值，供决定是否加到 -j16（写进报告）。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
LOG=$J/dl/batch_reconfig_par.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== 重配置批次（-j8 激进并行） $(date '+%F %H:%M:%S') ==="

# 自匹配陷阱：模式与自身 cmdline 同形会让 pgrep 永远为真。
done_marker() { grep -q 'BATCH_BUILD_NEXT_DONE' "$J/dl/batch_build_next.log" 2>/dev/null; }
compiler_alive() { pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1; }
# 关键：flag 探针与本批次**等的是同一个标记**（批次1 结束），若不同时串行化会一起醒来
# （探针 12 GB + 本批 8 个作业 = 新的 OOM 配方）。所以本批必须排在探针之后。
probe_done() { grep -q 'PROBE_ARM_DONE' "$J/dl/jit_flags_probe.log" 2>/dev/null; }
t0=$(date +%s)
while ! done_marker || compiler_alive || ! probe_done; do
  if [ $(( $(date +%s) - t0 )) -gt 21600 ]; then echo "TIMEOUT 等批次1/探针 (>6h)"; exit 2; fi
  sleep 30
done
echo "--- 批次1 已结束 $(date +%H:%M:%S) ---"
grep -E '^\- ' "$C/M_a5b_verify.md" 2>/dev/null | head -4

PATCHES=("$C/A3_spark_head_geometry.diff" "$C/A3_spark_headwise_gate.diff" \
         "$C/A3_spark_gelu_mul.diff" "$C/E7_s50_regression_sketch.diff")
resolve() {
  local f=$1
  patch -p1 --dry-run -d "$R" < "$f" >/dev/null 2>&1 && { echo "$f"; return 0; }
  local t=/tmp/jp_$(basename "$f"); tr -d '\r' < "$f" > "$t"
  patch -p1 --dry-run -d "$R" < "$t" >/dev/null 2>&1 && { echo "$t"; return 0; }
  return 1
}
for f in "${PATCHES[@]}"; do
  [ -f "$f" ] || { echo "缺 $f"; exit 3; }
  r=$(resolve "$f") || { echo "$(basename "$f") 打不上，整批放弃"; exit 3; }
  patch -p1 -b -d "$R" < "$r" || { echo "落补丁失败 $r"; exit 4; }
  echo "  applied: $(basename "$r")"
done

cd "$R/build" || exit 5
echo "--- cmake 重新配置 $(date +%H:%M:%S) ---"
cmake -S "$R" -B "$R/build" > /tmp/par_reconfig.log 2>&1
rc=$?; tail -3 /tmp/par_reconfig.log; [ $rc -ne 0 ] && { echo "reconfigure 失败"; exit 6; }

# 激进并行：per-job 只算 3 GB、预留 1 GB、上限 8 个作业、真危险才干预
echo "--- 激进并行构建（PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 MIN_FREE_GB=0.5）---"
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 MIN_FREE_GB=0.5 \
  bash "$J/_par_build.sh" ninfer ninfer-serve
pbrc=$?
echo "  par_build rc=$pbrc"
tail -3 "$J/dl/par_build.log" 2>/dev/null

echo "--- ctest（A3 的验收：不是'编译过'就算）---"
ctest --test-dir "$R/build" -R 'gelu_mul|sigmoid_mul|prepare_masked_block|context_store' \
      --output-on-failure > /tmp/par_ctest.log 2>&1
echo "  ctest rc=$?"
tail -10 /tmp/par_ctest.log

{
  echo ""
  echo "## 重配置批次（-j8 激进并行） $(date '+%F %H:%M')"
  echo "- 落的补丁：A3 head_geometry / headwise_gate / gelu_mul + E7/S50 回归"
  echo "- 构建方式：PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 MIN_FREE_GB=0.5（允许溢到 SSD swap）"
  echo "- par_build rc=$pbrc"
  echo "- 墙钟与 swap 观察（供决定是否加到 -j16）："
  grep -E 'make .* -j|用时|MemAvailable=' "$J/dl/par_build.log" 2>/dev/null | tail -12 | sed 's/^/    /'
  echo "- ctest 结论："
  tail -6 /tmp/par_ctest.log | sed 's/^/    /'
} >> "$C/M_sparkops_verify.md"
echo "=== 批次结束 $(date '+%F %H:%M:%S') ==="
echo BATCH_RECONFIG_PAR_DONE
