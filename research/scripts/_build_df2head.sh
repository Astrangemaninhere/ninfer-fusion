#!/bin/bash
# 增量重编：dflash2 草稿头接线。
# 仓库没有头文件依赖跟踪 ⇒ 必须 touch 吃到这些头的 TU（否则 make 认为是空的增量）。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/build_df2head.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== build df2head $(date '+%F %H:%M:%S') ==="

cd $R || exit 2
echo "--- touch TUs that include the edited headers ---"
for f in src/targets/qwen3_6_27b/impl/variant.cpp \
         apps/cli/options.cpp apps/cli/main.cpp \
         src/serve/serve_options.cpp src/serve/request_log.cpp; do
  if [ -f "$f" ]; then touch "$f"; echo "  touched $f"; else echo "  MISSING $f"; fi
done

echo "--- make ninfer -j2 ---"
cd $R/build || exit 3
start=$(date +%s)
make ninfer -j2 2>&1 | tail -60
rc=${PIPESTATUS[0]}
end=$(date +%s)
echo "make rc=$rc elapsed=$(( (end-start)/60 ))m$(( (end-start)%60 ))s"
ls -l --time-style=+%m-%d_%H:%M $R/build/apps/ninfer 2>/dev/null
if [ "$rc" -eq 0 ]; then echo "BUILD_DF2HEAD_OK"; else echo "BUILD_DF2HEAD_FAIL"; fi
