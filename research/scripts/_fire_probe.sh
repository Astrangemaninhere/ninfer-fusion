#!/bin/bash
# 决定性一枪：NINFER_DF2DBG=1 读 extent 与逐列明细
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
cd "$R/build" || exit 1
echo "二进制: $(stat -c '%y' "$BIN" | cut -c1-19)  $(stat -c %s "$BIN") bytes"
echo
echo '=== dflash2 K=7 + 探针 ==='
NINFER_DF2DBG=1 timeout 900 "$BIN" "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" \
  --prompt "$P" --max-new 32 --max-context 4096 --no-thinking --greedy \
  --print-token-ids --spec dflash2 --draft-tokens 7 > /home/user/probe_df2.log 2>&1
echo "rc=$?"
echo '--- 探针行（前 25 条）---'
grep -iE 'df2dbg|df2 probe|extent|col=|verify|draft=|argmax' /home/user/probe_df2.log | head -25 | cut -c1-200
echo
echo '--- 接受率/剖面 ---'
grep -oE 'dflash2 acceptance rate +[0-9.]+%' /home/user/probe_df2.log | head -1
grep -oE 'accepted by pos +[0-9,]+' /home/user/probe_df2.log | tail -1
grep -oE 'dflash2 rounds +[0-9]+' /home/user/probe_df2.log | head -1
echo
echo '--- 若探针无输出，列出含 T3 的所有行 ---'
grep -inE 'T3|probe|debug' /home/user/probe_df2.log | head -10 | cut -c1-180
