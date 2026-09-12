#!/bin/bash
# 串行判决：① BF16 头 artifact 能否跑通（解锁后应给出 head/embed 量化误差的非循环参考）
#           ② NINFER_DF2SCORES 分数矩阵 dump（离线树材料复算）
# 两者串行执行，避免多实例争抢 32GB 显存。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
OLD=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
NEW=/home/user/models/qwen3_8_27b_nvfp4_dflash2_bf16head.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/serial_verdict.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || exit 3
echo "=== 串行判决 起 $(date '+%m-%d %H:%M:%S') ==="

echo "--- ① BF16 头 artifact：先短生成冒烟（plain）---"
timeout 600 ./apps/ninfer "$NEW" --prompt "$P" --max-new 16 --max-context 1024 \
  --no-thinking --greedy > $J/dl/sv_bf16head_plain.log 2>&1
echo "  rc=$?  （0=跑通，非0=仍缺内核）"
grep -m1 -E 'error' $J/dl/sv_bf16head_plain.log | head -c 200; echo
grep -m1 'decode speed' $J/dl/sv_bf16head_plain.log | sed 's/^/  /'

echo "--- ① b BF16 头 artifact：dflash2 接受率 ---"
timeout 600 ./apps/ninfer "$NEW" --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --spec dflash2 > $J/dl/sv_bf16head_d2.log 2>&1
grep -m1 -E 'dflash2 acceptance rate|dflash2 acceptance length|dflash2 accepted by pos|decode speed' \
  $J/dl/sv_bf16head_d2.log | sed 's/^/  /'

echo "--- ① c 对照：旧 FP8 头 artifact dflash2（同 prompt）---"
timeout 600 ./apps/ninfer "$OLD" --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --spec dflash2 > $J/dl/sv_fp8head_d2.log 2>&1
grep -m1 -E 'dflash2 acceptance rate|dflash2 acceptance length|dflash2 accepted by pos|decode speed' \
  $J/dl/sv_fp8head_d2.log | sed 's/^/  /'

echo "--- ② 分数矩阵 dump（NINFER_DF2SCORES，需 --no-cuda-graph）---"
rm -f $J/dl/df2scores_*.bin 2>/dev/null
NINFER_DF2SCORES=1 NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$OLD" --prompt "$P" \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --no-cuda-graph \
  > $J/dl/sv_scores.log 2>&1
echo "  rc=$?"
echo "  dump 文件数: $(ls $J/dl/df2scores_scores_*.bin 2>/dev/null | wc -l)"
ls -l $J/dl/df2scores_scores_1.bin 2>/dev/null | awk '{print "  scores_1 字节 =", $5}'
grep -m2 'df2scores] n=' $J/dl/sv_scores.log | sed 's/^/  /'
grep -m1 'dflash2 accepted by pos' $J/dl/sv_scores.log | sed 's/^/  /'
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo SERIAL_VERDICT_DONE
