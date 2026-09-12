#!/bin/bash
# KV 精度 x dflash2 接受率（修正 kv-dtype 取值）+ MTP 健康配置对照
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/kv_ab3.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }
echo "=== KV 精度 x 接受率 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-18s %-9s %-22s %-11s %s\n" arm 接受率 位置剖面 解码 轮数
run() {
  local label="$1"; shift
  timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy \
    "$@" > $J/dl/kv3_$label.log 2>&1
  local a p sp rd kind
  a=$(grep -m1E 'dflash2|mtp|dflash ' $J/dl/kv3_$label.log | grep -m1E 'acceptance rate' | grep -oE '[0-9.]+%')
  p=$(grep -m1E 'dflash2|mtp|dflash ' $J/dl/kv3_$label.log | grep -m1E 'accepted by pos' | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' $J/dl/kv3_$label.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m1E 'dflash2 rounds|mtp rounds|dflash rounds' $J/dl/kv3_$label.log | grep -oE '[0-9]+')
  printf "  %-18s %-9s %-22s %-11s %s\n" "$label" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}"
}
run d2_default       --spec dflash2
run d2_kvbf16        --spec dflash2 --kv-dtype bf16
run d2_kvbf16_w1     --spec dflash2 --kv-dtype bf16 --draft-tokens 1
run d2_kvbf16_w3     --spec dflash2 --kv-dtype bf16 --draft-tokens 3
run d2_kvint8        --spec dflash2 --kv-dtype int8
run mtp_healthy      --spec mtp --draft-tokens 3 --lm-head-draft
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo KV_AB3_DONE
