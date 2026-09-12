#!/bin/bash
A=/home/user/ninfer-fusion
B=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "=== md5 compare A(wsl) vs B(win repo):"
for f in src/ops/launcher/swa.cu \
         src/ops/kernel/bidirectional_gqa_attention.cuh \
         src/ops/wrapper/swa.cpp \
         src/targets/qwen3_6/impl/runtime/dflash2_impl.h \
         src/targets/qwen3_6/impl/runtime/dflash_impl.h \
         src/targets/qwen3_6_27b/impl/config.h \
         src/targets/qwen3_6/impl/runtime/program_impl.h; do
  a=$(md5sum "$A/$f" 2>/dev/null | cut -d' ' -f1)
  b=$(md5sum "$B/$f" 2>/dev/null | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then s=SAME; else s=DIFF; fi
  printf "%-64s %s A=%.12s B=%.12s\n" "$f" "$s" "$a" "$b"
done
echo "=== git A:"
cd "$A" && git rev-parse --is-inside-work-tree 2>&1 | head -1
git log --oneline -3 2>&1 | head -5
echo "--- A status (porcelain, top 20):"
git status --porcelain 2>&1 | head -20
echo "=== git B:"
cd "$B" && git rev-parse --is-inside-work-tree 2>&1 | head -1
git log --oneline -3 2>&1 | head -5
echo "--- B status:"
git status --porcelain 2>&1 | head -20
echo "=== B log for swa files (last 5 commits touching them):"
cd "$B" && git log --oneline -5 -- src/ops/launcher/swa.cu 2>&1
git log --oneline -5 -- src/ops/kernel/bidirectional_gqa_attention.cuh 2>&1
git log --oneline -5 -- src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>&1
