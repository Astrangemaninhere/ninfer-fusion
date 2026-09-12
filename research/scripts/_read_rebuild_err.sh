#!/bin/bash
echo "=== reb_ninfer_1.log：编的是哪个 TU、第一个错误是什么 ==="
grep -nE 'Building|error:|In file included|from ' /tmp/reb_ninfer_1.log 2>/dev/null | head -24 | cut -c1-170
echo
echo "=== 错误计数与类型分布 ==="
grep -c 'error:' /tmp/reb_ninfer_1.log 2>/dev/null | sed 's/^/  errors: /'
grep -oE 'error: .{0,60}' /tmp/reb_ninfer_1.log 2>/dev/null | sort | uniq -c | sort -rn | head -8
