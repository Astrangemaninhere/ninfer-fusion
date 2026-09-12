#!/bin/bash
set -u
cd /home/user/ninfer-fusion || exit 1
for f in \
  src/targets/qwen3_6/impl/runtime/dflash_impl.h \
  src/ops/common/math.cuh \
  src/targets/qwen3_6/impl/runtime/text_context_impl.h \
  src/ops/kernel/mtp_round.cuh \
  src/ops/wrapper/gqa_attention.cpp ; do
  cr=$(awk '/\r$/{c++} END{print c+0}' "$f")
  tot=$(wc -l < "$f")
  kind=LF
  [ "$cr" -gt 0 ] && kind=CRLF
  printf '  %-62s CR=%-6s lines=%-6s %s\n' "$f" "$cr" "$tot" "$kind"
done
