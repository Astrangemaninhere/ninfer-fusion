#!/bin/bash
# sync changed files from windows repo to wsl build tree
SRC=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
DST=/home/user/ninfer-fusion
while IFS= read -r f; do
  cp "$SRC/$f" "$DST/$f" && echo "OK $f"
done < "$SRC/../_changed.txt" 2>/dev/null || while IFS= read -r f; do
  cp "$SRC/$f" "$DST/$f" && echo "OK $f"
done < /mnt/c/Users/User/Documents/ziqinzhang/_changed.txt
echo SYNC_DONE
