#!/bin/bash
# dflash2 对拍：① plain 给出目标的贪心续写 ② NINFER_DF2DBG=1 给出逐轮草稿/verify
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
cd "$R/build" || exit 1

echo "=== 二进制时间戳（应晚于 07:37）==="
ls -la --time-style=+%H:%M:%S "$BIN" | cut -c25-70
echo "编译状态: nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  错误=$(grep -cE ' error:' /mnt/c/Users/User/Documents/ziqinzhang/dl/par_build.log 2>/dev/null)"

echo
echo '=== ① plain：目标的贪心续写 ==='
timeout 600 "$BIN" "$M/qwen3_8_27b_nvfp4.ninfer" --prompt "$P" --max-new 32 --max-context 4096 \
    --no-thinking --greedy --print-token-ids > /home/user/ab_plain.log 2>&1
grep -oE '^tokens +generated ids.*' /home/user/ab_plain.log | cut -c1-160 | sed 's/^/  /'

echo
echo '=== ② dflash2 K=7 + 探针 ==='
NINFER_DF2DBG=1 timeout 900 "$BIN" "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" --prompt "$P" --max-new 32 \
    --max-context 4096 --no-thinking --greedy --print-token-ids --spec dflash2 --draft-tokens 7 \
    > /home/user/ab_df2.log 2>&1
echo "  rc=$?"
echo '  --- 探针输出（前 20 行）---'
grep -E 'T3 probe|DF2DBG|draft|extent|verify' /home/user/ab_df2.log | head -20 | cut -c1-150 | sed 's/^/    /'
echo '  --- 接受率/剖面 ---'
grep -oE 'dflash2 acceptance rate +[0-9.]+%' /home/user/ab_df2.log | head -1 | sed 's/^/    /'
grep -oE 'accepted by pos +[0-9,]+' /home/user/ab_df2.log | tail -1 | sed 's/^/    /'
echo '  --- 本次生成的 ids ---'
grep -oE '^tokens +generated ids.*' /home/user/ab_df2.log | cut -c1-160 | sed 's/^/    /'
date +%H:%M:%S
