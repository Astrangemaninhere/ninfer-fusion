#!/bin/bash
# print one ASCII line with the current pack count + newest file mtime
n=$(ls /home/user/bench/hs_cache_topk 2>/dev/null | wc -l)
newest=$(ls -t /home/user/bench/hs_cache_topk 2>/dev/null | head -1)
echo "count=$n newest=$newest now=$(date +%H:%M:%S)"
