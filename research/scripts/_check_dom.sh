#!/bin/bash
echo "=== $(date +%H:%M:%S) ==="
echo "nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)"
echo "--- dom_build.log 尾 ---"
tail -4 /tmp/dom_build.log 2>/dev/null | cut -c1-125
echo "--- par_build 日志尾 ---"
tail -4 /mnt/c/Users/User/Documents/ziqinzhang/dl/par_build.log 2>/dev/null | cut -c1-125
echo "--- domain_exp.log 尾 ---"
tail -6 /mnt/c/Users/User/Documents/ziqinzhang/dl/domain_exp.log 2>/dev/null | cut -c1-125
echo "--- 是否还在跑 ---"
pgrep -f 'btd[.]sh' >/dev/null && echo "  building" || echo "  done"
