#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== launcher/dflash2_selector.h 10-25（cat -A 看空格）==='
awk 'NR>=10 && NR<=25 {printf "%4d|%s\n", NR, $0}' "$R/src/ops/launcher/dflash2_selector.h" | cat -A | cut -c1-160
echo
echo '=== launcher/dflash2_selector.cu 11-22（签名处）==='
awk 'NR>=11 && NR<=22 {printf "%4d|%s\n", NR, $0}' "$R/src/ops/launcher/dflash2_selector.cu" | cat -A | cut -c1-160
