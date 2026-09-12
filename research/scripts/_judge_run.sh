#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer

echo "=== 二进制/编译状态 ==="
ls -la --time-style=+%m-%d_%H:%M:%S "$BIN" | cut -c25-75
echo "nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  错误=$(grep -cE ' error:' $J/dl/par_build.log 2>/dev/null)"
MT=$(stat -c %Y "$BIN" 2>/dev/null); NOW=$(date +%s)
if [ "$(( NOW - MT ))" -gt 3600 ] && [ "$(pgrep -c -x nvcc 2>/dev/null || echo 0)" != "0" ]; then
  echo "  => 编译仍在进行，探针还没进二进制；本次只报状态"
  exit 0
fi
echo "  => 二进制是新的，开跑判据"
cd "$R/build" || exit 1

echo
echo '=== 判据①：探针读 extent（dflash2 K=7）==='
NINFER_DF2DBG=1 timeout 900 "$BIN" "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" --prompt "$P" --max-new 32 \
  --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash2 --draft-tokens 7 \
  > /home/user/judge_df2.log 2>&1
echo "  rc=$?"
grep -iE 'df2dbg|in_extent|out_extent|valid=|col=|verify=|draft=|argmax' /home/user/judge_df2.log | head -24 | cut -c1-155

echo
echo '=== 判据②：目标贪心续写（plain，取前 32 个 id 供对拍）==='
timeout 600 "$BIN" "$M/qwen3_8_27b_nvfp4.ninfer" --prompt "$P" --max-new 32 --max-context 4096 \
  --no-thinking --greedy --print-token-ids > /home/user/judge_plain.log 2>&1
grep -oE '^tokens +generated ids.*' /home/user/judge_plain.log | cut -c1-170

echo
echo '=== 接受率/剖面（同一次 dflash2 跑）==='
grep -oE 'dflash2 acceptance rate +[0-9.]+%' /home/user/judge_df2.log | head -1
grep -oE 'accepted by pos +[0-9,]+' /home/user/judge_df2.log | tail -1
date +%H:%M:%S
