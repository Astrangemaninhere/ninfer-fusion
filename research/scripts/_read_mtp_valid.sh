#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 1) mtp_round.cuh 头部：next / next_extents 的来源与语义 ==="
sed -n '1,36p' "$R/src/ops/kernel/mtp_round.cuh" | cat -n | cut -c1-150
echo
echo "=== 2) mtp_impl.h：valid 的用途（传给谁、怎么掩码） ==="
sed -n '108,125p;170,205p' "$R/src/targets/qwen3_6/impl/runtime/mtp_impl.h" | cat -n | cut -c1-150
