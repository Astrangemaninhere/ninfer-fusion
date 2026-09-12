#!/bin/bash
# 层号修正实验的复测：dspark d7 接受率（对照：改前 7.63% / 位置 [17,1,0,0,0,0,0]）
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
cd "$R/build" || exit 1
echo "二进制: $(stat -c '%y' "$BIN" | cut -c1-19)"
echo
run() {
  local tag=$1; shift
  local out=/home/user/layerexp_$tag.log
  timeout 900 "$BIN" "$@" > "$out" 2>&1
  local pos acc tpr rnd
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -1 | sed 's/.*pos *//')
  acc=$(grep -oE 'dflash acceptance rate +[0-9.]+%' "$out" | head -1)
  rnd=$(grep -oE 'dflash rounds +[0-9]+' "$out" | head -1)
  printf '  %-16s %-34s pos=[%s] %s\n' "$tag" "$acc" "${pos:-无}" "$rnd"
}
echo '=== dspark（层号改为 {2,15,30,45,58} 之后）==='
run dsp_k7 "$M/qwen3_8_27b_nvfp4_dspark.ninfer" --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash --draft-tokens 7
run dsp_k3 "$M/qwen3_8_27b_nvfp4_dspark.ninfer" --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash --draft-tokens 3
run dsp_k1 "$M/qwen3_8_27b_nvfp4_dspark.ninfer" --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash --draft-tokens 1
echo
echo '=== 对照：dflash2（层号未改，应仍 ~4.97%）==='
run df2_k7 "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash2 --draft-tokens 7
echo
echo '=== 基线参照（改前实测）==='
echo '  dspark d7:  accept 7.63%  pos=[17,1,0,0,0,0,0]'
echo '  dspark k1:  accept 22.08% pos=[17]'
echo '  dflash2 d7: accept 4.97%  pos=[8,0,0,0,0,0,0]'
echo '  健康区间（代码注释）: 21-28%'
date +%H:%M:%S
