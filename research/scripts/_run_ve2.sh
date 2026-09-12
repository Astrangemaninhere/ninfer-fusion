#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_verify_equivalence2.sh" && echo SYNTAX_OK || exit 1
echo "=== 前台跑 v2（能看到全部输出） ==="
bash "$J/_verify_equivalence2.sh" 2>&1 | tail -22 | cut -c1-200
