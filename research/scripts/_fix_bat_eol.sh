#!/bin/bash
# 把包装 .bat 转成 CRLF（cmd 对 LF 解析会崩），然后后台启动
F=/mnt/c/Users/User/Documents/ziqinzhang/_run_vllm_msvc.bat
sed -i 's/\r$//' "$F"
sed -i 's/$/\r/' "$F"
echo "--- 换行符检查（应显示 CRLF）---"
file "$F"
head -3 "$F" | cat -A | head -3
