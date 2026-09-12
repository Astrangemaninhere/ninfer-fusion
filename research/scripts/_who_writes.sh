#!/bin/bash
# 查谁在写草稿文件；只保留一个写者，必要时用 curl -C - 续传到完成
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
F=$J/data/draft_dflash2_ref/model.safetensors
echo "=== 1) 当前大小 ==="
stat -c '%s bytes  mtime=%y' "$F" 2>/dev/null
sleep 10
stat -c '%s bytes  mtime=%y' "$F" 2>/dev/null
echo
echo "=== 2) 谁在跑（含 cmdline） ==="
ps -eo pid,etimes,args | grep -iE 'curl|snapshot_download|huggingface|hf_transfer|redo2' | grep -v grep | cut -c1-150
echo
echo "=== 3) 相关脚本是否还活着 ==="
pgrep -af 'redo2.sh|dl_dflash2' 2>/dev/null | cut -c1-120
