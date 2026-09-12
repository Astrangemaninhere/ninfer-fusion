#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== working-dir top level (dirs only) ==="
find "$J" -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | head -40
echo
echo "=== repo tools/ tree (2 levels) ==="
find "$J/ninfer-fusion-repo/tools" -maxdepth 2 -type d 2>/dev/null | sed "s#$J/ninfer-fusion-repo/##" | sort | head -30
echo
echo "=== any gui/studio/i18n files in the repo ==="
find "$J/ninfer-fusion-repo" -maxdepth 3 \( -iname '*gui*' -o -iname '*studio*' -o -iname '*i18n*' -o -iname '*locale*' -o -iname '*translat*' \) 2>/dev/null | sed "s#$J/ninfer-fusion-repo/##" | head -20
echo
echo "=== 'spark' + x2/2.5 in TODO / docs (context) ==="
grep -rniE 'spark[ -]?x?2\.5|sparkx|spark x' "$J/_TODO.md" "$J"/_*.md "$J/_collab"/*.md 2>/dev/null | head -10 | cut -c1-160
echo "--- model import list mentions ---"
grep -rniE '导入|import' "$J/_TODO.md" 2>/dev/null | head -14 | cut -c1-150
