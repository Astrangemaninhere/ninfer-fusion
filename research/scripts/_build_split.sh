#!/bin/bash
# 落地后验证 + touch 我改过的头文件的包含者 + 后台编译
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/build_split.log
export PATH=/home/user/.local/bin:$PATH
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 拆分落地后验证 + 编译 $(date '+%F %H:%M:%S') ==="

echo "--- 1) 拆分文件现状 ---"
wc -l $R/src/ops/launcher/gqa_attention_decode.cu \
      $R/src/ops/launcher/gqa_attention_decode_partial.cuh \
      $R/src/ops/launcher/gqa_attention_decode_smallt.cu \
      $R/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu \
      $R/src/ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh 2>/dev/null
echo "--- 2) 旧 TU 是否已 include 新头 ---"
grep -c 'gqa_attention_decode_partial.cuh' $R/src/ops/launcher/gqa_attention_decode.cu
grep -c 'nvfp4_w4a4_tma_arms.cuh' $R/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu
echo "--- 3) 被迁移的符号在旧 TU 里应已不存在 ---"
for sym in launch_tc_partial_ single_row_batch_view NINFER_GQA_SMALL_T_DISPATCH; do
  echo "  $sym in old .cu = $(grep -c "$sym" $R/src/ops/launcher/gqa_attention_decode.cu)"
done
echo "--- 4) CMake 新条目 ---"
grep -nE 'gqa_attention_decode_smallt|nvfp4_w4a4_tma_(attn|gdn|mlp|residual)' $R/src/CMakeLists.txt
echo "--- 5) 新产物无 BF16 partial ---"
echo "  partial.cuh: $(grep -cE 'static_cast<(const )?__nv_bfloat16\*>\s*\(\s*partial_acc\.data' $R/src/ops/launcher/gqa_attention_decode_partial.cuh)"
echo "  smallt.cu : $(grep -cE 'static_cast<(const )?__nv_bfloat16\*>\s*\(\s*partial_acc\.data' $R/src/ops/launcher/gqa_attention_decode_smallt.cu)"

echo "--- 6) touch 我改过的 4 个头文件的包含者 ---"
for h in layouts_impl.h program_impl.h kv_calibration.h text_context_impl.h; do
  n=0
  for inc in $(grep -rl "$h" $R/src $R/apps 2>/dev/null | grep -v '\.orig' | sed "s|$R/||"); do
    case "$inc" in *.h|*.cuh|*.orig) continue;; esac
    touch "$R/$inc"; n=$((n+1))
  done
  echo "  $h -> touched $n TU(s)"
done

echo "--- 7) 编译（-j3，拆分后单 TU 峰值内存下降） ---"
cd "$R/build" || exit 4
start=$(date +%s)
make ninfer -j3 > /tmp/mk_split.log 2>&1
rc=$?
echo "make rc=$rc elapsed=$(( ($(date +%s)-start)/60 ))m"
tail -6 /tmp/mk_split.log
if [ "$rc" -eq 0 ]; then
  ls -l --time-style=+%H:%M $R/build/apps/ninfer | cut -c1-60
  echo BUILD_SPLIT_OK
else
  echo BUILD_SPLIT_FAIL
  grep -nE 'error|Error' /tmp/mk_split.log | head -12
fi
