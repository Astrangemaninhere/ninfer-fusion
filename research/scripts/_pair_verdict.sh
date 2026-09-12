#!/bin/bash
# 判决实验：pair 在任何位置有用吗？
#  W=1（只需 s=0）与 W=3 各扫 scale ∈ {0, 1}：若 W=1 下 scale=0 明显低于 28% ⇒ pair 在 s=0 有用；
#  再看 W=3 的 pos1/pos2 在 scale=0 vs 1 下是否变化 ⇒ pair 在 s≥1 是否承载链条。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/pair_verdict.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }
echo "=== pair 判决实验 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-14s %-9s %-22s %-11s %s\n" 臂 接受率 位置剖面 解码 轮数
run() {
  local label="$1"; shift
  timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy \
    "$@" > $J/dl/pv_$label.log 2>&1
  local a p sp rd
  a=$(grep -m 1 'dflash2 acceptance rate' $J/dl/pv_$label.log | grep -oE '[0-9.]+%')
  p=$(grep -m 1 'dflash2 accepted by pos' $J/dl/pv_$label.log | sed 's/.*pos *//')
  sp=$(grep -m 1 'decode speed' $J/dl/pv_$label.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m 1 'dflash2 rounds' $J/dl/pv_$label.log | grep -oE '[0-9]+')
  printf "  %-14s %-9s %-22s %-11s %s\n" "$label" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}"
}
NINFER_DF2_PAIR_SCALE=0 run w1_s0   --spec dflash2 --draft-tokens 1
NINFER_DF2_PAIR_SCALE=1 run w1_s1   --spec dflash2 --draft-tokens 1
NINFER_DF2_PAIR_SCALE=0 run w3_s0   --spec dflash2 --draft-tokens 3
NINFER_DF2_PAIR_SCALE=1 run w3_s1   --spec dflash2 --draft-tokens 3
NINFER_DF2_PAIR_SCALE=0 run w7_s0   --spec dflash2
NINFER_DF2_PAIR_SCALE=1 run w7_s1   --spec dflash2
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo PAIR_VERDICT_DONE
