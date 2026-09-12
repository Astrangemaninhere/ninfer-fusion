#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== md5 & mtime of .orig vs live:"
for f in src/targets/qwen3_6/impl/runtime/dflash2_impl.h src/targets/qwen3_6/impl/runtime/dflash_impl.h src/targets/qwen3_6/impl/runtime/layouts_impl.h src/CMakeLists.txt; do
  echo "-- $f"
  ls -la --time-style=long-iso "$R/$f" "$R/$f.orig" 2>/dev/null
done
echo
echo "=== diff dflash2_impl.h vs .orig (unified, 3 lines ctx):"
diff -u "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h.orig" "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h" | head -200
echo
echo "=== diff dflash_impl.h vs .orig:"
diff -u "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h.orig" "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | head -200
