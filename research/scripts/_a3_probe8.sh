#!/bin/bash
T=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
cd $T
echo "############ md5sum -c against A3 baseline snapshot (live_md5.txt)"
md5sum -c $C/a3_scratch/live_md5.txt 2>&1 | grep -v ': OK$' | head -40
echo "------------ total lines / OK count"
md5sum -c $C/a3_scratch/live_md5.txt 2>&1 | grep -c ': OK$'
echo "############ A3 gelu_mul new files exist?"
for f in include/ninfer/ops/gelu_mul.h src/ops/launcher/gelu_and_mul.h src/ops/wrapper/gelu_mul.cpp src/ops/kernel/gelu_and_mul.cuh src/ops/launcher/gelu_and_mul.cu tests/ops/test_gelu_mul.cpp; do
  if [ -e "$f" ]; then echo "EXISTS  $f  ($(wc -l < $f) lines)"; else echo "absent  $f"; fi
done
echo "############ tests/ops dir listing (sigmoid/gelu/silu + orig + .rej)"
ls -la tests/ops/ | grep -Ei 'sigmoid|gelu|silu'
echo "############ any .rej anywhere in tree?"
find $T -name '*.rej' -not -path '*/build/*' 2>/dev/null | head -20
echo "############ .orig files in tree (non-build)"
find $T -name '*.orig' -not -path '*/build/*' 2>/dev/null | head -40
echo "############ tests/CMakeLists.txt mentions"
grep -n -E 'sigmoid_mul|gelu_mul|silu_mul' tests/CMakeLists.txt
echo "############ src/CMakeLists.txt mentions"
grep -n -E 'sigmoid|gelu' src/CMakeLists.txt
echo "############ -Werror in build flags?"
grep -rn 'Werror\|Wall\|Wextra' CMakeLists.txt src/CMakeLists.txt tests/CMakeLists.txt 2>/dev/null | head -20
grep -rn 'CMAKE_CXX_FLAGS' CMakeLists.txt | head -20
echo "############ g++/nvcc flags in build dir cache (if present)"
grep -m5 -o 'CMAKE_CXX_FLAGS[^=]*=[^"]*' build/CMakeCache.txt 2>/dev/null | head -10
echo "############ wrapper.orig vs wrapper (bytes at 23:06)"
stat -c '%y %s %n' src/ops/wrapper/sigmoid_mul.cpp src/ops/wrapper/sigmoid_mul.cpp.orig tests/ops/test_sigmoid_mul.cpp tests/ops/test_sigmoid_mul.cpp.orig 2>/dev/null
