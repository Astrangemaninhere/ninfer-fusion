#!/bin/bash
# 准备 A5b 的回退版（A5b = 让 bf16 分支的 attention_valid 保持 width；回退 = 恢复降回 k）
# 目的：用户指出"开头你修了对齐的问题，但现在问题越来越严重" ⇒ A5b 可能有害。
# 本次只做 CPU 侧：确认当前状态 + 反向应用 A5b + 重编，等 GPU 空出再测。
set -u
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h
export PATH="/home/user/.local/bin:$PATH"

echo '=== 当前 dflash_impl.h 里 attention_valid 相关行 ==='
grep -n 'set_i32_scalar(attention_valid\|Restore the draft block\|width, state.execution' "$F" | head -8 | cut -c1-140
echo
echo '=== 反向应用 A5b 补丁（dry-run 先行）==='
cp "$F" /home/user/dflash_impl.bak_before_a5b_revert
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/A5b_attention_valid_width.diff
[ -f "$D" ] || D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/A5b_attention_valid_width.diff
echo "  补丁: $D"
if [ -f "$D" ]; then
  tr -d '\r' < "$D" > /tmp/a5b.diff
  if patch -R -p1 --dry-run -d "$R" < /tmp/a5b.diff >/dev/null 2>&1; then
    patch -R -p1 -d "$R" < /tmp/a5b.diff && echo "  已反向应用（A5b 已回退）"
  else
    echo "  反向 dry-run 失败 —— 说明 A5b 不在树里或已被改"
    patch -R -p1 --dry-run -d "$R" < /tmp/a5b.diff 2>&1 | head -5
  fi
else
  echo "  找不到 A5b 补丁文件"
fi
echo
echo '=== 回退后确认 ==='
grep -n 'set_i32_scalar(attention_valid' "$F" | head -4 | cut -c1-140
