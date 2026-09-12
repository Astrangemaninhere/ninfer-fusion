#!/bin/bash
# 多 prompt × (dflash2 / mtp) 接受率判决扫描：检验 4.81% 是否 prompt 特异
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
LOG=$J/dl/prompt_scan.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }

ZH='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
EN='Write a short paragraph explaining how a transformer attention head computes its output.'
CODE='Write a Python function that merges two sorted lists in linear time, with a docstring.'

echo "=== 多 prompt 判决扫描 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-10s %-9s %-9s %-22s %-11s %s\n" prompt backend 接受率 位置剖面 解码 轮数

runl() { # $1 tag  $2 后端标签  $3... 命令参数
  local tag="$1"; local bl="$2"; shift 2
  timeout 900 ./apps/ninfer "$M" --max-new 96 --max-context 4096 --no-thinking \
    "$@" > $J/dl/ps_$tag.log 2>&1
  local a p sp rd
  a=$(grep -m 1 -E 'dflash2 acceptance rate|mtp acceptance rate' $J/dl/ps_$tag.log | grep -oE '[0-9.]+%')
  p=$(grep -m 1 -E 'dflash2 accepted by pos|mtp accepted by pos' $J/dl/ps_$tag.log | sed 's/.*pos *//')
  sp=$(grep -m 1 'decode speed' $J/dl/ps_$tag.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m 1 -E 'dflash2 rounds|mtp rounds' $J/dl/ps_$tag.log | grep -oE '[0-9]+')
  printf "  %-10s %-9s %-9s %-22s %-11s %s\n" "$tag" "$bl" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}"
}

runl zh_d2    dflash2 --prompt "$ZH" --greedy --spec dflash2
runl zh_mtp   mtp     --prompt "$ZH" --greedy --spec mtp --draft-tokens 3 --lm-head-draft
runl en_d2    dflash2 --prompt "$EN" --greedy --spec dflash2
runl en_mtp   mtp     --prompt "$EN" --greedy --spec mtp --draft-tokens 3 --lm-head-draft
runl code_d2  dflash2 --prompt "$CODE" --greedy --spec dflash2
runl code_mtp mtp     --prompt "$CODE" --greedy --spec mtp --draft-tokens 3 --lm-head-draft
runl fx_d2    dflash2 --messages /home/user/prefill-probe-msg.json --spec dflash2
runl fx_mtp   mtp     --messages /home/user/prefill-probe-msg.json --spec mtp --draft-tokens 3 --lm-head-draft
runl zh_t1_d2 dflash2 --prompt "$ZH" --temperature 1.0 --seed 7 --spec dflash2
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo PROMPT_SCAN_DONE
