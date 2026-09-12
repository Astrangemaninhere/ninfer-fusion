#!/bin/bash
# 用 sed 收紧启动器的错误检测正则（去掉会误判的裸 'error:'）
F=/mnt/c/Users/User/Documents/ziqinzhang/_vllm_launch.py
cp "$F" "$F.bak"
# 该行原文：  if re.search(r"ModuleNotFoundError|ValueError|RuntimeError|Traceback|error:|not supported|No module", t):
sed -i 's/ModuleNotFoundError|ValueError|RuntimeError|Traceback|error:|not supported|No module/Traceback \\(most recent call last\\)|ModuleNotFoundError|ImportError|api_server\\.py: error:/' "$F"
echo "--- 修改后的 re.search 行 ---"
grep -n 're.search' "$F"
echo "--- 差异 ---"
diff "$F.bak" "$F" || true
rm -f "$F.bak"
