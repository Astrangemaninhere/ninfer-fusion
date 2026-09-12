#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
L=$J/dl/par_build.log
echo '=== par_build.log 最后 35 行（含 21:54 之后的失败现场）==='
tail -35 "$L" | cut -c1-140
echo
echo '=== 21:54 之后的所有非进度行 ==='
awk '/21:5[4-9]:|2[2-3]:[0-9][0-9]:/' "$L" | grep -vE 'Building (CXX|CUDA) object|Built target' | tail -25 | cut -c1-150
echo
echo '=== 树是否被 batch1 改过（A5b / S52 是否已落）==='
R=/home/user/ninfer-fusion
grep -c 'source_column_offset = 1' $R/src/targets/qwen3_6/impl/runtime/dflash_impl.h
grep -c 'Restore the draft block' $R/src/targets/qwen3_6/impl/runtime/dflash_impl.h
grep -c 'block_drafts\|draft window' $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo '  ^ dflash_impl 的 offset=1 / 旧注释残留 / dflash2 相关'
grep -c 'kDFlashDecodeMaximumDrafts' $R/src/product/speculative_options.h 2>/dev/null
echo '  ^ S52 是否已落（>0 = 已落）'
ls -la --time-style=+%H:%M "$J/dl/batch_build_next.log" 2>/dev/null | cut -c25-90
tail -5 "$J/dl/batch_build_next.log" 2>/dev/null | cut -c1-130
echo
echo '=== batch1 的标记是否存在 ==='
grep -l 'BATCH_BUILD_NEXT_DONE' "$J/dl/"*.log 2>/dev/null
