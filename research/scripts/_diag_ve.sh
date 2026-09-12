#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 1) dflash2 用例为何 rc=1 ==="
tail -8 /home/user/ve_dflash2_a.log 2>/dev/null | cut -c1-170
echo
echo "=== 2) --print-token-ids 的真实输出格式（源码） ==="
grep -rn 'print_token_ids\|print-token-ids' "$R/apps/cli/"*.cpp "$R/apps/cli/"*.h 2>/dev/null | head -6 | cut -c1-150
grep -rn -A6 'if (cli.print_token_ids\|print_token_ids)' "$R/apps/cli/main.cpp" 2>/dev/null | head -20 | cut -c1-150
echo
echo "=== 3) 一次 plain 运行的原始输出（看 token id 行长什么样） ==="
grep -nE '[0-9]{3,}' /home/user/ve_plain_a.log 2>/dev/null | head -8 | cut -c1-160
echo "  --- 全文尾部 ---"
tail -12 /home/user/ve_plain_a.log 2>/dev/null | cut -c1-160
