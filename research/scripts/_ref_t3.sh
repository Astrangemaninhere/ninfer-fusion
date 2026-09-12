#!/bin/bash
echo "=== reference impl copies on disk ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 \( -iname 'qwen3_dflash*.py' -o -iname '*dflash*.py' -o -iname 'speculator*.py' -o -iname 'spec_generate*' \) 2>/dev/null | head -30
echo
echo "=== dirs named dflash/vllm ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 3 -type d \( -iname '*dflash*' -o -iname 'vllm*' \) 2>/dev/null | head -20
echo
echo "=== train_dspark.py / aeon trainer locations ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 \( -iname 'train_dspark*.py' -o -iname 'aeon-train_head.py' -o -iname 'train_dflash*.py' \) 2>/dev/null | head -20
echo
echo "=== WSL home scripts ==="
ls /home/user/*.py 2>/dev/null | head -20
echo
echo "=== _collab dir: list all files mentioning spec_generate or dflash2 columns ==="
grep -rln 'spec_generate\|source_column_offset\|block_drafts' /mnt/c/Users/User/Documents/ziqinzhang/_collab 2>/dev/null | head -20
