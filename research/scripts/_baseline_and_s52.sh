#!/bin/bash
# ① S52 现在能否干净落库（A5b 已落，检查是否冲突）
# ② 给"当前二进制"建立稳定基线：dspark d7 与 dflash2 auto 各 3 次，取中位
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
export PATH="/home/user/.local/bin:$PATH"

echo '=== ① S52 dry-run（正/反两个方向）==='
cp "$C/E9_s52_dflash2_k_slice.diff" /tmp/s52_raw.diff
patch -p1 --dry-run -d "$R" < /tmp/s52_raw.diff > /tmp/s52f.txt 2>&1
echo "  正向 rc=$?  $(tail -2 /tmp/s52f.txt | tr '\n' ' ' | cut -c1-120)"
patch -p1 --dry-run -R -d "$R" < /tmp/s52_raw.diff > /tmp/s52r.txt 2>&1
echo "  反向 rc=$?  $(head -3 /tmp/s52r.txt | tr '\n' ' ' | cut -c1-120)"
echo

echo '=== ② 基线（各 3 次）==='
cd "$R/build" || exit 1
one() {  # tag model args...
  local tag=$1 model=$2; shift 2
  local log=/home/user/base_${tag}_$3.log 2>/dev/null || true
  local out=/home/user/base_$tag.log
  timeout 900 "$BIN" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$out" 2>&1
  local pos spd ids
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -1 | sed 's/.*pos *//')
  spd=$(grep -oE 'decode speed +[0-9.]+' "$out" | head -1 | grep -oE '[0-9.]+')
  ids=$(grep -oE '^tokens +generated ids.*' "$out" | head -1 | md5sum | cut -c1-8)
  echo "  $tag  pos=[${pos:-无}]  decode=${spd:-?}  ids_md5=$ids"
}
for i in 1 2 3; do one "dspark7_r$i" qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7; done
for i in 1 2 3; do one "df2auto_r$i" qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto; done
date +%H:%M:%S
