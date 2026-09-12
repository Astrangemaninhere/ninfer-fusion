#!/bin/bash
# 草稿宽度扫描：如果位置 ≥1 从不被接受，那么更宽的草稿只是白烧算力。
# 用同一个 prompt、贪心、96 token，跑 dspark 的 1/3/7 档 + dflash2 对照，记录位置剖面与速度。
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
export PATH="/home/user/.local/bin:$PATH"
cd "$R/build" || exit 1

run() {  # tag model spec-width...
  local tag=$1 model=$2; shift 2
  local log=/home/user/width_$tag.log
  timeout 900 "$BIN" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  local rc=$?
  local pos spd
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  spd=$(grep -oE 'decode speed +[0-9.]+' "$log" | head -1 | grep -oE '[0-9.]+')
  printf '  %-18s rc=%s  pos=[%s]  decode=%s tok/s\n' "$tag" "$rc" "${pos:-无}" "${spd:-?}"
}

echo '=== dspark（草稿宽度扫描）==='
run dspark_d1 "$M" x 2>/dev/null || true
run dspark_d1 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 1
run dspark_d3 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 3
run dspark_d7 qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7
echo '=== dflash2（对照）==='
run df2_auto qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto
run df2_k3 qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto --draft-tokens 3
echo '=== 基线（plain）==='
run plain qwen3_8_27b_nvfp4.ninfer --spec none
echo
echo '=== 位置剖面的含义 ==='
echo '  pos=[a0,a1,...] 第 i 位被接受的次数；若 a1.. 全 0 => 更宽的草稿宽度不产生任何收益'
date +%H:%M:%S
