#!/bin/bash
# 带 MTP 逐步 logits 插桩的运行（需 --no-cuda-graph：dump 内有同步）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
rm -f "$J"/mtplg_*.bin
echo "=== MTP logits dump 起 $(date '+%H:%M:%S')  宿主: $(uptime | sed 's/.*load average/load/') ==="
NINFER_MTPLOG=1 timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 32 --max-context 4096 \
  --no-thinking --greedy --spec mtp --draft-tokens 3 --lm-head-draft --no-cuda-graph \
  > "$J/mtplg_run.log" 2>&1
echo "rc=$?"
echo "dump 文件: logits=$(ls "$J"/mtplg_logits_*.bin 2>/dev/null | wc -l) tokens=$(ls "$J"/mtplg_tokens_*.bin 2>/dev/null | wc -l)"
ls -l "$J"/mtplg_logits_1.bin 2>/dev/null | awk '{print "  logits_1 字节 =", $5}'
grep -m3 'mtplog\] n=' "$J/mtplg_run.log" | sed 's/^/  /'
grep -E 'decode speed|acceptance length|acceptance rate' "$J/mtplg_run.log" | sed 's/^/  /'
echo "=== 完成 $(date '+%H:%M:%S') ==="
echo MTPLOG_RUN_DONE
