#!/bin/bash
# 调大 WSL 内存+swap 以容纳 vLLM 参考跑（先备份，可逆）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
CFG=/mnt/c/Users/User/.wslconfig
BAK=$J/dl/wslconfig.bak.$(date +%H%M)

echo "=== 1) 现状 ==="
if [ -f "$CFG" ]; then echo "--- 现有 .wslconfig ---"; cat "$CFG"; cp -f "$CFG" "$BAK"; echo "已备份到 $BAK"; else echo "(无 .wslconfig，将新建)"; fi

echo "=== 2) 写入新配置（memory 24GB / swap 32GB / 关闭自动回收） ==="
cat > "$CFG" <<'EOF'
[wsl2]
memory=24GB
swap=32GB
swapFile=C:\\wsl-swap.vhdx
pageReporting=false
autoMemoryReclaim=disabled
EOF
cat "$CFG"
echo "=== 3) 重启 WSL 使配置生效 ==="
exit 0
