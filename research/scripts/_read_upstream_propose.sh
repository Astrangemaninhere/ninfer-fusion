#!/bin/bash
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream/src/targets/qwen3_6/impl/runtime/dflash_impl.h
echo '=== 上游 dflash_impl.h: 205-335（branch helpers + propose_dflash2_batch）==='
awk 'NR>=205 && NR<=335 {printf "%4d| %s\n", NR, $0}' "$U" | cut -c1-155
