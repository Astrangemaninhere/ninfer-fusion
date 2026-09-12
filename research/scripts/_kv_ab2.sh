#!/bin/bash
# 决定性对照：同一 CLI、同一 prompt、只换 KV 精度；另含 draft-tokens=3（对照健康配置）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/kv_ab.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }
echo "=== KV 精度 x dflash2 接受率 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-16s %-9s %-22s %-11s %s\n" arm 接受率 位置剖面 解码 轮数
run() {
  local label="$1"; shift
  timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy \
    "$@" > $J/dl/kvab_$label.log 2>&1
  local a p sp rd
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/kvab_$label.log | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/kvab_$label.log | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' $J/dl/kvab_$label.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m1 'dflash2 rounds' $J/dl/kvab_$label.log | grep -oE '[0-9]+')
  printf "  %-16s %-9s %-22s %-11s %s\n" "$label" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}"
}
run default     --spec dflash2
run kv_bf16     --spec dflash2 --kv-dtype bfloat16
run kv_bf16_w3  --spec dflash2 --kv-dtype bfloat16 --draft-tokens 3
run kv_nvfp4    --spec dflash2 --kv-dtype nvfp4
run kv_bf16_w1  --spec dflash2 --kv-dtype bfloat16 --draft-tokens 1
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo KV_AB_DONE
