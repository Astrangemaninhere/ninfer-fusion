#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== who consumes serve_params.json? ==="
grep -rn 'serve_params' "$R/tools" "$R"/../ninfer-fusion-repo/tools 2>/dev/null | head -10 | cut -c1-140
echo "--- existing registry file? ---"
find "$R" "$J" -maxdepth 3 -name 'serve_params*.json' 2>/dev/null | head -5
echo
echo "=== the GUI entry point ==="
find "$R" "$J" -maxdepth 3 -iname 'ninfer-gui*' -o -maxdepth 3 -iname '*gui*.py' 2>/dev/null | grep -v __pycache__ | head -12
echo
echo "=== serve_gui.py: how the command line is built + does it read a registry? ==="
grep -nE 'def |subprocess|cmd|argv|json|params|PARAMS|form|request|/api/|app\.|route' "$R/tools/gui/serve_gui.py" | head -30 | cut -c1-140
