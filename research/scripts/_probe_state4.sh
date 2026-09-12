#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_todo_s50.py"
echo "--- patchA build tail ---"
tail -3 "$J/dl/patchA_build.log" | cut -c1-120
echo "--- make progress ---"
grep -oE '^\[[ 0-9]+%\][^"]*' /tmp/pa_make_1.log | tail -1
for p in $(pgrep -f bin/nvcc); do echo "  nvcc elapsed $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo "--- watcher ---"
tail -1 "$J/dl/postbuild_measure.log" | cut -c1-100
echo "--- E8/E9 progress (their dirs) ---"
ls -lt "$J/_collab"/E8_* "$J/_collab"/E9_* 2>/dev/null | head -6 | awk '{print "  ", $5, $NF}'
echo "--- upstream clone present ---"
ls -d /mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream >/dev/null 2>&1 && echo "  yes" || echo "  no"
