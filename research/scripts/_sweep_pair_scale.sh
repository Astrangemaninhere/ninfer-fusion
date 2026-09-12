#!/bin/bash
# 边项权重扫描：1.0（契约）-> 0（等价于修复前的 unary-only 行为）以及中间值。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/pair_scale.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== pair scale sweep $(date '+%F %H:%M:%S') ==="
python3 $J/_apply_pair_scale.py || { echo PATCH_FAILED; exit 2; }
cd $R/build || exit 3
make ninfer -j2 2>&1 | tail -6
rc=${PIPESTATUS[0]}
echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 1; }

P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
for s in 1.0 0.5 0.25 0.1 0.0; do
  log=/home/user/ps_$s.log
  NINFER_DF2_PAIR_SCALE=$s timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 \
      --max-context 4096 --no-thinking --greedy --spec dflash2 > "$log" 2>&1
  pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  acc=$(grep -oE 'acceptance rate[[:space:]]+[0-9.]+%' "$log" | tail -1 | sed 's/.* //')
  al=$(grep -oE 'acceptance length[[:space:]]+[0-9.]+' "$log" | tail -1 | sed 's/.* //')
  echo "  scale=$s  rate=$acc  AL=$al  pos=[$pos]"
done
echo PAIR_SCALE_DONE
