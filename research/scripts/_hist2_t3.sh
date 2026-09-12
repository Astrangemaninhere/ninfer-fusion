#!/bin/bash
cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo || exit 1
echo "=== does mirror have dflash2 impl? ==="
ls -la src/targets/qwen3_6/impl/runtime/ 2>/dev/null | grep -i dflash
echo
echo "=== history of source_column_offset in dflash_impl.h ==="
git log --oneline -S 'source_column_offset' -- src/targets/qwen3_6/impl/runtime/dflash_impl.h 2>&1 | head -10
echo
echo "=== history: commits touching dflash2_impl.h ==="
git log --oneline -- src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>&1 | head -20
echo
echo "=== show the offset line in mirror HEAD ==="
git show HEAD:src/targets/qwen3_6/impl/runtime/dflash2_impl.h 2>/dev/null | grep -n 'element_bytes\|source_column\|hidden \*' | head -12
echo
echo "=== diff of the region HEAD vs working tree (mirror) ==="
git diff HEAD -- src/targets/qwen3_6/impl/runtime/dflash2_impl.h src/targets/qwen3_6/impl/runtime/dflash_impl.h 2>&1 | head -60
