#!/bin/bash
echo "=== _orig_quarantine tree ==="
find /home/user/ninfer-fusion/_orig_quarantine -maxdepth 3 | head -40
echo
echo "=== quarantine dflash files content check ==="
for f in $(find /home/user/ninfer-fusion/_orig_quarantine -name 'dflash*impl.h' -o -name 'dflash2_impl.h' | head -5); do
  echo "--- $f"
  grep -n 'source_column_offset\|hidden \* element_bytes\|element_bytes' "$f" | head -10
done
echo
echo "=== backup dirs in _collab/build ==="
find /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/backup /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged2 -maxdepth 2 2>/dev/null | head -40
echo
echo "=== ninfer-fusion-repo git log ==="
cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo 2>/dev/null && git log --oneline -20 2>&1 | head -25
