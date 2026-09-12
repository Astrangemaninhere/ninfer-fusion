#!/bin/bash
# MTP 效率实测：k=1..5 的 tok/s / AL / 轮数 ⇒ 推每轮耗时与流量分解
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/mtp_eff.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
echo "=== MTP 效率实测 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-8s %-9s %-9s %-22s %-11s %-6s %s\n" k 接受率 AL tok/s 轮数 ms/轮 位置剖面
run() {
  local k="$1"; shift
  timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec mtp "$@" > "$J/mtp_k$k.log" 2>&1
  local a al sp rd p
  a=$(grep -m1 'mtp acceptance rate' "$J/mtp_k$k.log" | grep -oE '[0-9.]+%')
  al=$(grep -m1 'mtp acceptance length' "$J/mtp_k$k.log" | grep -oE '[0-9.]+')
  sp=$(grep -m1 'decode speed' "$J/mtp_k$k.log" | grep -oE '[0-9.]+')
  rd=$(grep -m1 'mtp rounds' "$J/mtp_k$k.log" | grep -oE '[0-9]+')
  p=$(grep -m1 'mtp accepted by pos' "$J/mtp_k$k.log" | sed 's/.*pos *//')
  local ms=?
  if [ -n "${sp:-}" ] && [ -n "${al:-}" ] && [ "${sp}" != "0" ]; then
    ms=$(awk -v s="$sp" -v a="$al" 'BEGIN{printf "%.1f", 1000.0*a/s}')
  fi
  printf "  %-8s %-9s %-9s %-22s %-11s %-6s %s\n" "$k" "${a:-?}" "${al:-?}" "${sp:-?}" "${rd:-?}" "$ms" "${p:-?}"
}
run 1 --draft-tokens 1 --lm-head-draft
run 2 --draft-tokens 2 --lm-head-draft
run 3 --draft-tokens 3 --lm-head-draft
run 4 --draft-tokens 4 --lm-head-draft
run 5 --draft-tokens 5 --lm-head-draft
echo
echo "=== 参考：理论权重流量下界（5090D 约 1.79 TB/s）==="
awk 'BEGIN{
  w=22.66; bw=1.79;
  printf "  verify 单轮读全模型: %.2f GB -> %.1f ms\n", w, 1000*w/bw;
  for(k=1;k<=5;k++){
    head=k*2.5; tot=w+head;
    printf "  k=%d: 草稿头 +%.1f GB -> 合计 %.2f GB -> %.1f ms/轮; 若 AL 与该 k 匹配则 tok/s 上界≈%.0f\n",
      k, head, tot, 1000*tot/bw, 1000*(1+k*0.7)/(1000*tot/bw)*1000/1000;
  }
  printf "  （AL 上界粗估 1+0.7k 仅示意；实际见上表）\n";
}'
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo MTP_EFF_DONE
