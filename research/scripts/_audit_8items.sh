#!/bin/bash
# 对 _board_append8.md 里那 8 条未落地项逐条取证（每条只回答"做了没有"）
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$R" || exit 1

echo '### 1) N1 --kv-bit-budget 落地了吗（apps 侧解析）'
grep -rn 'kv-bit-budget' apps/ src/product/ 2>/dev/null | head -5 | cut -c1-120 || true
echo '   apps/cli/options.cpp 那一行现状:'
sed -n '144p' apps/cli/options.cpp 2>/dev/null | cut -c1-120
echo '   A_n1_patch.diff 是否还打得上（打得上=还没落）:'
tr -d '\r' < "$J/_collab/A_n1_patch.diff" > /tmp/n1.diff 2>/dev/null && \
  patch -p1 --dry-run < /tmp/n1.diff >/dev/null 2>&1 && echo '     可打 => 未落地' || echo '     打不上 => 已落或已变'
echo
echo '### 2) N2 --max-cold-pages CLI 标志'
grep -rn 'max-cold-pages' apps/ src/ 2>/dev/null | head -4 | cut -c1-120 || echo '   (无)'
echo
echo '### 3) N3 运行时校准闭环 --recalibrate'
grep -rn 'recalibrate' apps/ src/ 2>/dev/null | head -4 | cut -c1-120 || echo '   (无)'
echo '### 3b) N5/W13 权重卸载 P1 钩子'
grep -rn 'w13\|weight_offload\|offload_hook' src/ 2>/dev/null | head -3 | cut -c1-120 || echo '   (无)'
echo '### 3c) N6 ngram 真表 gather'
grep -rn 'ngram' src/ apps/ 2>/dev/null | head -4 | cut -c1-120 || echo '   (无)'
echo
echo '### 4) N7 FlashNext 权重是否全量在盘'
du -sh /home/user/models/*qwen4* /home/user/xh* 2>/dev/null | head -4
ls -la /home/user/models/ 2>/dev/null | grep -iE 'qwen4|flashnext|exp' | head -5 | cut -c25-110
echo
echo '### 5) U7 Muse page-fill 补丁在树上吗'
ls -la "$J/_collab/M_muse_pagefill_patch.diff" 2>/dev/null | cut -c25-95
tr -d '\r' < "$J/_collab/M_muse_pagefill_patch.diff" > /tmp/u7.diff 2>/dev/null && {
  patch -p1 --dry-run < /tmp/u7.diff >/dev/null 2>&1 && echo '   可打 => 未落地'
  patch -p1 --dry-run < /tmp/u7.diff 2>&1 | grep -qi 'previously applied\|Reversed' && echo '   反向命中 => 已落地';
}
echo
echo '### 7) 103 sliding_window_tokens 有没有赋值点'
grep -rn 'sliding_window_tokens' src/ 2>/dev/null | head -8 | cut -c1-125
echo
echo '### 8) 其他解析点 perplexity 侧'
sed -n '115p' apps/perplexity/main.cpp 2>/dev/null | cut -c1-120
echo
echo '### 顺带：本轮 acceptance 修复链的状态'
ps -ef | grep -E 'pfv2[.]sh|batch_next[.]sh|reconfig_next[.]sh|probe_arm[.]sh' | grep -v grep | awk '{print "   链:", $2, $NF}'
date +%H:%M:%S
