#!/bin/bash
# MVP-0（零代码）：列的边际成本 C 与链的深度收益 R
# 判据：R/C > 1 ⇒ 额外列在链上已经净赚，树才值得投；否则树必须先证明"列的边际成本非线性"。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
LOG=$J/dl/mvp0_width.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || exit 3
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
echo "=== MVP-0 宽度/深度探针 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-16s %-10s %-9s %-24s %-12s %s\n" 臂 列数 AL 位置剖面 解码 轮数
run() { # $1 label
  local label="$1"; shift
  timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy "$@" > $J/dl/m0_$label.log 2>&1
  local al p sp rd cols sk
  sk=$(grep -m 1 -oE '(dflash2|mtp) ' $J/dl/m0_$label.log | head -1 | tr -d ' ')
  al=$(grep -m 1 -E 'dflash2 acceptance length|mtp acceptance length' $J/dl/m0_$label.log | grep -oE '[0-9.]+ tok/round')
  p=$(grep -m 1 -E 'dflash2 accepted by pos|mtp accepted by pos' $J/dl/m0_$label.log | sed 's/.*pos *//')
  sp=$(grep -m 1 'decode speed' $J/dl/m0_$label.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m 1 -E 'dflash2 rounds|mtp rounds' $J/dl/m0_$label.log | grep -oE '[0-9]+')
  case "$label" in
    d2_w8)  cols=8;;  d2_w16) cols=16;;
    mtp_w4) cols=4;;  mtp_w8) cols=8;;
    *) cols=?;;
  esac
  printf "  %-16s %-10s %-9s %-24s %-12s %s\n" "$label" "$cols" "${al:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}"
}
run d2_w8   --spec dflash2
run d2_w16  --spec dflash2 --draft-tokens 15
run mtp_w4  --spec mtp --draft-tokens 3 --lm-head-draft
run mtp_w8  --spec mtp --draft-tokens 7 --lm-head-draft
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo MVP0_DONE
