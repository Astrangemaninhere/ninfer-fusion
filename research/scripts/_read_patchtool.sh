#!/bin/bash
# 读 patch_dflash2.py 的映射与写出逻辑（求逆的前提）+ 轮询 (A)
J=/mnt/c/Users/User/Documents/ziqinzhang
F=$J/ninfer-fusion-repo/tools/convert/qwen3_8_27b/patch_dflash2.py
echo "=== 文件规模 ==="
wc -l "$F" 2>/dev/null; stat -c '%s bytes  %y' "$F" 2>/dev/null
echo
echo "=== 结构与关键函数 ==="
grep -nE '^def |^class |^[A-Z_]+ *=|mapping|TENSOR|NAMES|lookup|def write|def main|dflash2/|namespace' "$F" 2>/dev/null | head -40
echo
echo "=== 张量名/命名空间相关行 ==="
grep -nE "'dflash2|dflash2/|context_key|context_value|query_key_value|hidden_norm|attention_conv|mlp_conv|selector|codebook" "$F" 2>/dev/null | head -30
