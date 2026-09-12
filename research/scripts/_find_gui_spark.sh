#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
F=$J/ninfer-fusion-repo
echo "=== 'spark' mentions in docs (case-insensitive) ==="
grep -rniE 'spark' "$J/_TODO.md" 2>/dev/null | head -12 | cut -c1-150
echo "--- in _collab ---"
grep -rlniE 'spark' "$J/_collab" 2>/dev/null | head -8
echo "--- in the repo (paths only) ---"
grep -rliE 'spark' "$F/src" "$F/tools" "$F/docs" 2>/dev/null | head -12
echo
echo "=== GUI / studio / i18n surface ==="
ls -d "$F"/*/ 2>/dev/null | head -20
grep -rniE 'gui|studio' "$J/_TODO.md" 2>/dev/null | head -10 | cut -c1-140
echo "--- translation/翻译 in TODO ---"
grep -rniE 'i18n|翻译|locale|zh-CN|translat' "$J/_TODO.md" 2>/dev/null | head -12 | cut -c1-140
echo
echo "=== gui-ish dirs in the working dir ==="
ls -d "$J"/*[Gg][Uu][Ii]* "$J"/*studio* "$J"/*[Ss]tudio* 2>/dev/null | head
