#!/bin/bash
# 查上游是否有 487f8977（"merge ... DFlash2 speculative decoding"），以及它与我们的树差什么
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo '=== 上游克隆是否存在 ==='
[ -d "$U" ] && echo "  在: $U" || echo "  无（需重新 clone）"
[ -d "$U" ] || exit 0
cd "$U" || exit 1
echo
echo '=== 上游里是否有 487f8977 ==='
git log --oneline -1 487f8977 2>/dev/null | cut -c1-120 || echo "  该 SHA 不在本地（需 fetch）"
echo
echo '=== 上游近期与 dflash2 相关的提交 ==='
git log --oneline --all --grep='dflash2' -i 2>/dev/null | head -20 | cut -c1-130
echo
echo '=== 上游 master/dev 的最新提交（看我们 fork 之后有没有新东西）==='
git log --oneline -8 2>/dev/null | cut -c1-130
echo
echo '=== 上游是否有 dflash2 相关文件 ==='
ls -1 vllm 2>/dev/null | head -3
find . -maxdepth 4 -iname '*dflash2*' -not -path './.git/*' 2>/dev/null | head -10
