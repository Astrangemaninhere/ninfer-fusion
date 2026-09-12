#!/bin/bash
# 先跑测试：① -j8 重编（含已落的 A5b/S52/A3×2/S3/E_s51/T4-A）② 立刻跑纯主机验收
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
LOG=$J/dl/tests_now.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== tests_now $(date '+%F %H:%M:%S') ==="
free -g | sed -n 2p

echo
echo "########## ① 重编（-j8，后台并行） ##########"
cd "$R/build" || exit 1
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
echo "  已发出重编请求；编译期间不动 GPU"

echo
echo "########## ② E 组：几何越界必须响亮失败（纯 CPU） ##########"
if [ -f "$C/E_geo_acceptance.sh" ]; then
  bash "$C/E_geo_acceptance.sh" 2>&1 | tail -6
else
  echo "  缺脚本"
fi

echo
echo "########## ③ S3：行标定表载荷判据（纯 CPU） ##########"
if [ -f "$C/build/S3_rowscale_payload_check.py" ]; then
  python3 "$C/build/S3_rowscale_payload_check.py" "$R/src/ops/kernel/gqa_isoquant_row_scale.cu" 2>&1 | tail -6
else
  echo "  缺脚本"
fi

echo
echo "########## ④ S3：几何闸门 2x2（纯 CPU） ##########"
if [ -f "$C/build/S3_rowscale_geom_test.sh" ]; then
  bash "$C/build/S3_rowscale_geom_test.sh" 2>&1 | tail -14
else
  echo "  缺脚本"
fi

echo
echo "########## ⑤ T4 Stage A：PLE 相位/计数器（host-only，无需构建） ##########"
SC=/tmp/t4_verify
rm -rf "$SC"; mkdir -p "$SC/src/ops/ple" "$SC/tools/archkit/specs" "$SC/third_party"
cd "$R" || exit 1
cp src/ops/ple/ple_layout.cpp src/ops/ple/ple_layout.h src/ops/ple/ple_table.h src/ops/ple/ple_table.cu "$SC/src/ops/ple/" 2>/dev/null
cp tools/ple_reference.py "$SC/tools/" 2>/dev/null
cp tools/archkit/specs/qwen4_exp_spec.json "$SC/tools/archkit/specs/" 2>/dev/null
cp -r third_party/nlohmann "$SC/third_party/" 2>/dev/null
cd "$SC" || exit 1
patch -p1 < "$C/build/T4_stage_a.diff" >/dev/null 2>&1 && patch -p1 < "$C/build/T4_stage_b.diff" >/dev/null 2>&1
echo "  patch rc=$?"
if g++ -std=c++20 -O1 -Wall -Wextra -I src -I third_party -DNINFER_SOURCE_DIR='"."' \
      tools/ple_ngram_window_test.cpp src/ops/ple/ple_layout.cpp -o /tmp/ple_ngram_window_test 2>/tmp/t4_gpp.log; then
  rm -rf /tmp/ple_fx
  NINFER_PLE_STATS=1 /tmp/ple_ngram_window_test /tmp/ple_fx > /tmp/ple.log 2>&1
  echo "  ok 行数: $(grep -c '^ok ' /tmp/ple.log)"
  grep 'ple stats line' /tmp/ple.log | head -2 | sed 's/^/    /'
else
  echo "  g++ 失败（见 /tmp/t4_gpp.log）:"; tail -5 /tmp/t4_gpp.log | sed 's/^/    /'
fi

echo
echo "=== tests_now 完 $(date '+%F %H:%M:%S')（编译仍在后台跑）==="
echo TESTS_NOW_DONE
