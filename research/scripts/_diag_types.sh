#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== Choice / Inspection 的定义（决定是就地构造还是加移动） ==="
grep -n 'struct Choice\|struct Inspection\|using Choice\|Choice choice' "$R/src/runtime/engine/resource_manager.h" | head -8 | cut -c1-120
echo "--- Choice 的成员（看它为什么不可拷贝） ---"
awk '/struct Choice/,/^    };/' "$R/src/runtime/engine/resource_manager.h" | head -30 | cat -n | sed 's/^/  /' | cut -c1-130
echo
echo "=== 是否已有 delete 的拷贝/移动 ==="
grep -n 'delete\|= default' "$R/src/runtime/engine/resource_manager.h" | head -12 | cut -c1-120
echo
echo "=== 出错的两处上下文（含后续使用点） ==="
grep -n 'head_inspection\|candidate_inspection' "$R/src/runtime/engine/engine_core.h" | head -20 | cut -c1-120
echo
echo "=== 引擎里 inspect_admission 的签名 ==="
grep -rn 'inspect_admission' "$R/src/runtime/engine/"*.h | head -5 | cut -c1-140
