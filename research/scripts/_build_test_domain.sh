#!/bin/bash
# 重编（含 token-domain 过滤）+ 复测 dflash2 / dspark
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
export PATH="/home/user/.local/bin:$PATH"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

echo '=== touch 包含者（无头依赖跟踪）==='
cd "$R" || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'dflash2_impl.h' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
N=$(printf '%s\n' "$T" | grep -c . || true)
echo "  touch $N"
[ -n "$T" ] && touch $T

echo '=== 重编 ==='
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve > /tmp/dom_build.log 2>&1
rc=$?
echo "  build rc=$rc"
grep -E ' error:' /tmp/dom_build.log | head -5 | cut -c1-170
tail -3 /tmp/dom_build.log | cut -c1-110
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" | awk '{print "  ninfer:", $6, $5}'
[ $rc -ne 0 ] && { echo '编译失败，不出结论'; exit 3; }

echo
echo '=== 复测 ==='
run() {  # tag model args...
  local tag=$1 model=$2; shift 2
  local out=/home/user/dom_$tag.log
  timeout 900 "$R/build/apps/ninfer" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$out" 2>&1
  printf '  %-16s %-34s pos=[%s]\n' "$tag" \
    "$(grep -oE 'dflash2? acceptance rate +[0-9.]+%' "$out" | head -n 1)" \
    "$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -n 1 | sed 's/.*pos *//')"
}
run df2_k7 qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 7
run dsp_k7 qwen3_8_27b_nvfp4_dspark.ninfer  --spec dflash --draft-tokens 7
echo
echo '=== 对照（加域过滤之前）==='
echo '  dflash2 K=7: 4.97%  pos=[8,0,0,0,0,0,0]'
echo '  dspark  K=7: 7.38%  pos=[18,0,0,0,0,0,0]'
echo '  参照健康值（N=7）: 62.6/34.9/19.1/9.6/4.7/1.5/0.6%'
date +%H:%M:%S
