#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
M=/home/user/models
BIN=$R/build/apps/ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

echo "=== 构建状态 $(date +%H:%M:%S) ==="
echo "  二进制: $(stat -c '%y %s' "$BIN" 2>/dev/null | cut -c1-19,21-32)"
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  ptxas=$(pgrep -c -x ptxas 2>/dev/null || echo 0)"
echo "  编排存活: $(pgrep -c -f '_par_build[.]sh' 2>/dev/null || echo 0)"
echo "  进度: $(grep -oE '\[[ 0-9]+%\]' $J/dl/par_build.log 2>/dev/null | tail -1)  错误: $(grep -cE ' error:' $J/dl/par_build.log 2>/dev/null)"
echo "  内存: $(free -g | sed -n 2p)  交换: $(free -g | sed -n 3p)"
echo "  交换使用峰值线索: $(grep -c 'swappiness' $J/dl/build_spill_ssd.log 2>/dev/null) 条 swappiness 记录"
tail -4 $J/dl/build_spill_ssd.log 2>/dev/null | cut -c1-120 | sed 's/^/    /'

MT=$(stat -c %Y "$BIN" 2>/dev/null || echo 0); NOW=$(date +%s)
AGE=$(( (NOW - MT) / 60 ))
echo "  二进制年龄: ${AGE} 分钟"
if [ "$AGE" -lt 90 ]; then
  echo
  echo '=== 打枪：NINFER_DF2DBG=1 读 extent ==='
  cd "$R/build" || exit 1
  NINFER_DF2DBG=1 timeout 900 "$BIN" "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" --prompt "$P" --max-new 32 \
    --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash2 --draft-tokens 7 \
    > /home/user/judge_df2.log 2>&1
  echo "  rc=$?"
  grep -iE 'df2dbg|in_extent|out_extent|valid=|col=|verify|draft=|argmax' /home/user/judge_df2.log | head -22 | cut -c1-155 | sed 's/^/    /'
  grep -oE 'dflash2 acceptance rate +[0-9.]+%' /home/user/judge_df2.log | head -1 | sed 's/^/  /'
  grep -oE 'accepted by pos +[0-9,]+' /home/user/judge_df2.log | tail -1 | sed 's/^/  /'
else
  echo "  => 二进制还旧（编译未落地），不打枪；下次再看"
fi
