#!/bin/bash
# T1 给的两个零代码改动假说，立刻验（GPU 空闲）：
#   R1  = NINFER_DFLASH_SVIP_THRESHOLD=0  -> 关掉 SVIP 的提前截断（T1 实测 drafted/rounds 只有 2.34~3.11，未截断是 6.93）
#   R2a = --draft-tokens 6                -> 验"引擎块宽比 checkpoint 的 block_size=7 宽一列"这条
# 基线（同一 prompt、贪心、96 token）：dspark d7 = [17,1,0,0,0,0,0] / ~29 tok/s；plain = 43.46 tok/s
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
cd "$R/build" || exit 1

run() {  # tag model env-and-args...
  local tag=$1 model=$2; shift 2
  local out=/home/user/r1r2_$tag.log
  timeout 900 "$BIN" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$out" 2>&1
  local rc=$? pos spd dr
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -1 | sed 's/.*pos *//')
  spd=$(grep -oE 'decode speed +[0-9.]+' "$out" | head -1 | grep -oE '[0-9.]+')
  dr=$(grep -oE 'drafted[^,]*|rounds [0-9]+' "$out" | tr '\n' ' ')
  printf '  %-20s rc=%s pos=[%s] decode=%s\n' "$tag" "$rc" "${pos:-无}" "${spd:-?}"
}

echo '=== 基线（复核）==='
run base_d7 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7
echo '=== R1：关掉 SVIP 截断 ==='
NINFER_DFLASH_SVIP_THRESHOLD=0 run r1_svip0 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7
echo '=== R2a：块宽假说（draft-tokens 6）==='
run r2a_k6 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 6
echo '=== R1+R2a 同时 ==='
NINFER_DFLASH_SVIP_THRESHOLD=0 run r1r2_k6 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 6
echo '=== 对照：dflash2 显式 K=3（已知最优档 36.14 tok/s / pos 23,5,0）==='
run df2_k3 qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 3
date +%H:%M:%S
