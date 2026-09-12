#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
for L in srv_fp8.out srv_fp8b.out srv_f7.out srv_f6.out; do
  f=$J/dl/$L
  [ -f "$f" ] || continue
  echo "########## $L"
  grep -inE 'swap|cpu offload|offload|pinned|pin_memory|host|GiB memory|kv cache|blocks' "$f" 2>/dev/null | head -20
  echo
done
