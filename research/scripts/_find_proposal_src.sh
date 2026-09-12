#!/bin/bash
# 运行时到底用哪些 artifact 对象做 dflash2 提案？（定点 grep，只读）
R=/home/user/ninfer-fusion
echo "=== 1) draft_head 字样的出现点 ==="
grep -rn 'draft_head' $R/src $R/include 2>/dev/null | grep -v '\.orig' | head -20
echo
echo "=== 2) dflash2/ 命名空间在代码里的读取点 ==="
grep -rn '"dflash2/' $R/src $R/include 2>/dev/null | grep -v '\.orig' | head -30
echo
echo "=== 3) candidate_selector / predecessor_codebook 的读取点 ==="
grep -rn 'candidate_selector\|predecessor_codebook\|successor_codebook' $R/src $R/include 2>/dev/null | grep -v '\.orig' | head -15
echo
echo "=== 4) 选择器/草稿头在 artifact 里的绑定名（binding 表） ==="
grep -rn 'dflash2/[a-z_]*' $R/src/targets/qwen3_6_27b/impl/*.cpp $R/src/targets/qwen3_6_27b/impl/**/*.cpp 2>/dev/null | grep -v '\.orig' | head -20
