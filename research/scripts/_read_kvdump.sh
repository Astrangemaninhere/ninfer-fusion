#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== kvdump 的写出与 D2H 拷贝写法 ==="
grep -n -B 4 -A 18 'inline void kvdump_write_file' $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h | head -30
echo
echo "=== kvdump 调用点（看它怎么拷 D2H） ==="
grep -n -B 3 -A 16 'kvdump_write_file(' $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h | sed -n '20,70p'
