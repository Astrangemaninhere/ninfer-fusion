#!/bin/bash
# 落库批（按 T2 台账的顺序）：A3 三联（拆分前必须落）-> S3 行标定表 -> E_s51 用例 -> T4 Stage A
# 之后 -j8 重编 + 纯主机验收（T4 的 26 项、E 的几何验收、S3 的载荷判据）
# 纪律：逐条 dry-run 门禁；任一打不上就停在该条并报告，不半途乱落；全部 -b 留 .orig。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
LOG=$J/dl/land_batch1.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== land_batch1 $(date '+%F %H:%M:%S') ==="

# 顺序（T2 台账）：A3 三联 -> S3 -> E_s51 -> T4-A。E_geo 与 S3 硬冲突，本条不落（S3 先）。
PATCHES=(
  "$C/A3_spark_head_geometry.diff"
  "$C/A3_spark_headwise_gate.diff"
  "$C/A3_spark_gelu_mul.diff"
  "$C/build/S3_rowscale_pool.diff"
  "$C/E_s51_silu_extreme_negative_test.diff"
  "$C/build/T4_stage_a.diff"
)

resolve() {  # 原样 / strip CR 两种都试
  local f=$1
  patch -p1 --dry-run -d "$R" < "$f" >/dev/null 2>&1 && { echo "$f"; return 0; }
  local t=/tmp/lb_$(basename "$f"); tr -d '\r' < "$f" > "$t"
  patch -p1 --dry-run -d "$R" < "$t" >/dev/null 2>&1 && { echo "$t"; return 0; }
  return 1
}

for f in "${PATCHES[@]}"; do
  if [ ! -f "$f" ]; then echo "缺文件 $f => 停止"; exit 3; fi
  r=$(resolve "$f") || { echo "打不上: $(basename "$f") => 停止（前面的已落，后面的未落）"; exit 3; }
  patch -p1 -b -d "$R" < "$r" || { echo "落库失败: $(basename "$f")"; exit 4; }
  echo "  applied: $(basename "$f")"
done
echo "--- 已落 6 条 ---"

# 无头文件依赖跟踪：touch 包含者（不新增文件 => 不需要 reconfigure）
cd "$R" || exit 5
TOUCH=$(grep -rl --include='*.cpp' --include='*.cu' -e 'gqa_attention_geometry' -e 'gqa_isoquant_row_scale' -e 'ops/ple/ple_' src/ 2>/dev/null | sort -u)
echo "--- touch $(echo "$TOUCH" | wc -l) 个包含者 ---"
[ -n "$TOUCH" ] && touch $TOUCH

echo "--- -j8 重编 $(date +%H:%M:%S) ---"
cd "$R/build" || exit 6
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash "$J/_par_build.sh" ninfer ninfer-serve
echo "  par_build rc=$?"
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" "$R/build/apps/ninfer-serve" 2>/dev/null | awk '{print "  ", $6, $5, $NF}'

# 纯主机验收（不依赖新二进制）
echo "--- 纯主机验收 $(date +%H:%M:%S) ---"
echo "[T4 Stage A]"; ( cd /tmp/t4_verify 2>/dev/null && ls tools/ple_ngram_window_test.cpp >/dev/null 2>&1 && echo "  （副本目录在，需按 T4 报告 §立即测试 重跑）" ) || echo "  未建副本，跳过"
echo "[E 几何验收]"; [ -f "$C/E_geo_acceptance.sh" ] && bash "$C/E_geo_acceptance.sh" 2>&1 | tail -3
echo "[S3 载荷判据]"; [ -f "$C/build/S3_rowscale_payload_check.py" ] && python3 "$C/build/S3_rowscale_payload_check.py" "$R/src/ops/kernel/gqa_isoquant_row_scale.cu" 2>&1 | tail -4
echo "=== land_batch1 done $(date '+%F %H:%M:%S') ==="
echo LAND_BATCH1_DONE
