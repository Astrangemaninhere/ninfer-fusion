#!/bin/bash
# 对比测试 + 接受率：旧二进制(18:08) vs 新二进制，dspark / dflash2 各档 + plain 基线
# 抓三样：① accepted by pos 剖面 ② 日志里的 accepted/rounds 百分比 ③ decode tok/s
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NEW=$R/build/apps/ninfer
OLD=/home/user/ninfer_pre_fix
cd "$R/build" || exit 1

echo "新二进制: $(stat -c '%y' "$NEW" 2>/dev/null | cut -c1-19)  $(stat -c %s "$NEW" 2>/dev/null)"
echo "旧二进制: $(stat -c '%y' "$OLD" 2>/dev/null | cut -c1-19)  $(stat -c %s "$OLD" 2>/dev/null)"
echo

run() {  # tag bin model args...
  local tag=$1 bin=$2 model=$3; shift 3
  local out=/home/user/cmp_$tag.log
  timeout 900 "$bin" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$out" 2>&1
  local rc=$? pos spd acc
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -1 | sed 's/.*pos *//')
  spd=$(grep -oE 'decode speed +[0-9.]+' "$out" | head -1 | grep -oE '[0-9.]+')
  acc=$(grep -oE 'accepted [0-9]+ / [0-9]+|[0-9.]+%|rounds [0-9]+' "$out" | tr '\n' ' ' | head -c 60)
  printf '  %-22s rc=%s decode=%-6s pos=[%s]  %s\n' "$tag" "$rc" "${spd:-?}" "${pos:-无}" "${acc:-}"
}

echo '===== dspark：旧 vs 新（A/B）====='
run dsp_d7_OLD  "$OLD" qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7
run dsp_d7_NEW  "$NEW" qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7
echo '===== dflash2：显式 K 扫描（新二进制）====='
run df2_k1_NEW  "$NEW" qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 1
run df2_k3_NEW  "$NEW" qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 3
run df2_k7_NEW  "$NEW" qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 7
run df2_k3_OLD  "$OLD" qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 3
echo '===== mtp3（回归对照：应逐位等价）====='
run mtp3_NEW "$NEW" qwen3_8_27b_nvfp4.ninfer --spec mtp --draft-tokens 3
run mtp3_OLD "$OLD" qwen3_8_27b_nvfp4.ninfer --spec mtp --draft-tokens 3
echo '===== plain 基线 ====='
run plain_NEW "$NEW" qwen3_8_27b_nvfp4.ninfer
echo
echo '===== 完整接受率行（若有）====='
for f in /home/user/cmp_*.log; do
  line=$(grep -oE '.*(rounds [0-9]+|accepted [0-9]+).*' "$f" 2>/dev/null | head -1 | cut -c1-110)
  [ -n "$line" ] && printf '  %-22s %s\n' "$(basename "$f" .log)" "$line"
done
date +%H:%M:%S
