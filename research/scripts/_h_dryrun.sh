#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$R" || exit 3
echo "=== U6 定义 ==="
grep -rnE 'U6[^0-9a-zA-Z]' $J/_collab/*.md $J/_board_append*.md $J/_TODO.md 2>/dev/null | head -8
echo
echo "=== 已备补丁 dry-run（当前树） ==="
for p in A_n1_patch.diff A_n1b_cold_pages.diff A_s30_budget_cold.diff A_s24_window_table.diff B_s33_ple_wiring.diff; do
  f=$J/_collab/$p
  if [ ! -f "$f" ]; then echo "  -- $p : 缺文件"; continue; fi
  files=$(grep -cE '^\+\+\+ b/' "$f")
  out=$(patch -p1 --dry-run < "$f" 2>&1)
  rc=$?
  bad=$(printf '%s\n' "$out" | grep -cE 'FAILED|Hunk #.* failed')
  echo "  -- $p : files=$files dry_rc=$rc 失败hunk=$bad"
  [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -E 'FAILED|failed|Reversed|can.t find' | head -4 | sed 's/^/       /'
done
