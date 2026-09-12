#!/bin/bash
G=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui
echo "=== 导入页的 tier 渲染/过滤逻辑 ==="
grep -n 'tier' "$G/model_import.py" | head -26 | cut -c1-140
echo
echo "=== covered 现在会不会被丢掉 ==="
grep -nE "== *'hook'|== *'new_op'|== *'post'|covered" "$G/model_import.py" | head -14 | cut -c1-140
