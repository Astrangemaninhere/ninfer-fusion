#!/bin/bash
# KV-dtype A/B: 默认(未显式指定) vs --kv-dtype bf16 vs int8。
# 若默认就是 int8，且 bf16 下接受率恢复，则此前所有结论都要在 bf16 下重判。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/kv_ab.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== kv-dtype A/B $(date '+%F %H:%M:%S') ==="

echo "--- 默认值来源 (代码) ---"
grep -rn "kv_cache" $R/src/product/options.h 2>/dev/null | head -5
grep -rn "KvCacheStorage kv_cache\|kv_cache *=" $R/src/product/*.h $R/apps/cli/*.cpp 2>/dev/null | head -8
echo "--- 模型侧默认 (artifact identity) ---"
echo "--- 运行日志里 kv 相关行 ---"
grep -iE 'kv' /home/user/lmh2_df2_full.log 2>/dev/null | head -12

P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd $R/build || exit 4
DF2=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
DSP=/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer

run() {  # tag model extra...
  local tag=$1 art=$2; shift 2
  local log=/home/user/kvab_$tag.log
  timeout 900 ./apps/ninfer "$art" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  local rc=$?
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local acc; acc=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  local ac;  ac=$(grep -oE 'spec_accepted=[0-9]+' "$log" | tail -1)
  local dr;  dr=$(grep -oE 'spec_drafted=[0-9]+' "$log" | tail -1)
  local kv;  kv=$(grep -oiE 'kv[_ -]?(dtype|storage|quant)[^,;]{0,24}' "$log" | head -1)
  echo "  [$tag] rc=$rc $ac $dr $acc pos=[$pos]  kv=[$kv]"
}

echo "--- dflash2 --spec dflash2 ---"
run df2_default "$DF2" --spec dflash2
run df2_bf16    "$DF2" --spec dflash2 --kv-dtype bf16
run df2_int8    "$DF2" --spec dflash2 --kv-dtype int8
echo "--- dspark K=7 ---"
run dsp_default "$DSP" --spec dflash --draft-tokens 7
run dsp_bf16    "$DSP" --spec dflash --draft-tokens 7 --kv-dtype bf16
echo KV_AB_DONE
