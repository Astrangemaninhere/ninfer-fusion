#!/bin/bash
# 对照结果里 accept 是空的（"accept - -> -"），p0 只有计数没有分母 => 无法判定。
# 查三件事：① 运行日志里到底有什么 ② 接受率到底该以什么标记出现 ③ OLD 基线是不是"差了好几个修复"
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== ① 一次运行的日志全文（dspark_NEW）==='
cat /home/user/pfv_dspark_NEW.log 2>/dev/null | cut -c1-150
echo
echo '=== ② 接受率/位置剖面该以什么标记出现（在仓库里找打印点）==='
grep -rn 'spec_accept_rate\|accepted by pos' /home/user/ninfer-fusion/src --include=*.h --include=*.cpp --include=*.cu 2>/dev/null | head -6 | cut -c1-135
echo
echo '=== ③ 两个快照是不是同一个二进制（若相同，OLD 就不是"修复前"）==='
md5sum /home/user/ninfer_pre_fix /home/user/ninfer_before_a5b /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null
ls -la --time-style=+%m-%d_%H:%M /home/user/ninfer_pre_fix /home/user/ninfer_before_a5b /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null | cut -c25-78
