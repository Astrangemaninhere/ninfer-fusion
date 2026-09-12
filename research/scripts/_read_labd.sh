#!/bin/bash
echo '=== 1Cat LABD 的补丁归档 ==='
ls -la --time-style=+%m-%d_%H:%M /home/user/bench/pr355.diff /home/user/bench/pr366.diff 2>/dev/null | cut -c25-95
ls -d /home/user/bench 2>/dev/null && ls -1 /home/user/bench | head -20
echo
echo '=== pr355 头部（LABD 引入）==='
[ -f /home/user/bench/pr355.diff ] && head -40 /home/user/bench/pr355.diff | cut -c1-150
echo
echo '=== pr366 里的 lookup.py（关键机制）==='
[ -f /home/user/bench/pr366.diff ] && grep -nE '^\+\+\+ |lookup|longest|candidate|q16|q8|DFLASH_TOKENS' /home/user/bench/pr366.diff | head -30 | cut -c1-150
echo
echo '=== 验收/报告里提到的关键开关 ==='
grep -rnE 'VLLM_DFLASH2_CHAIN|DFLASH_TOKENS|lookup' /mnt/c/Users/User/Documents/ziqinzhang/_labd_1cat_research.md 2>/dev/null | head -12 | cut -c1-155
