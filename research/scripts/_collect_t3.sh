#!/bin/bash
echo "=== who writes ids16 / vals16 ==="
grep -rln 'ids16' /mnt/c/Users/User/Documents/ziqinzhang --include=*.py --include=*.cpp --include=*.h --include=*.hpp --include=*.md --include=*.json --include=*.sh 2>/dev/null | head -30
echo
echo "=== who writes vals16 ==="
grep -rln 'vals16' /mnt/c/Users/User/Documents/ziqinzhang --include=*.py --include=*.cpp --include=*.h --include=*.md 2>/dev/null | head -30
echo
echo "=== tree-side teacher/collect === "
grep -rln 'collect_hs\|collect-hs\|teacher' /home/user/ninfer-fusion/tools /home/user/ninfer-fusion/src 2>/dev/null | head -30
echo
echo "=== data dir listing ==="
ls /mnt/c/Users/User/Documents/ziqinzhang/data/ 2>/dev/null | head -40
echo "=== df2pilot packs ==="
ls /mnt/c/Users/User/Documents/ziqinzhang/data/df2pilot/packs 2>/dev/null | head -10
ls /mnt/c/Users/User/Documents/ziqinzhang/data/df2pilot/ckpts 2>/dev/null | head -10
