#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 最新 make 日志尾部（完整错误上下文） ==="
tail -30 /tmp/reb_ninfer_1.log | cut -c1-170
echo
echo "=== 错误行（松散匹配） ==="
grep -nE 'rror|FAILED|fatal' /tmp/reb_ninfer_1.log | tail -12 | cut -c1-170
