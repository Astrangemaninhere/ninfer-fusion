#!/bin/bash
S=/mnt/c/Users/User/Documents/ziqinzhang/_land_s6.sh
echo "lines=$(wc -l < "$S")"
grep -n 'ARTIFACTS=' "$S"
grep -n 'ORIGS=' "$S"
grep -n 'CMAKE_PLAN=' "$S"
grep -n 'e8\|prefill' "$S" | grep '|' | head -25
