#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 编译状态 ==="
pgrep -x make >/dev/null && echo "make 在跑" || echo "make 已结束"
pgrep -c nvcc 2>/dev/null
grep -qE 'BUILD_SPLIT_OK|BUILD_SPLIT_FAIL' $J/dl/build_split.log 2>/dev/null && tail -4 $J/dl/build_split.log || tail -2 /tmp/mk_split.log 2>/dev/null
free -g | head -2
echo
echo "=== A2_draft_eval.py 的 CLI ==="
grep -nE 'add_argument|def main|argparse' $J/_collab/A2_draft_eval.py | head -30
