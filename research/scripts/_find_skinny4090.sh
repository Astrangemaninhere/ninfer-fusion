#!/bin/bash
echo '=== 目录名里含 skinny / 4090 的（全盘，含 Windows 侧）==='
find /home/user /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -type d \( -iname '*skinny*' -o -iname '*4090*' \) 2>/dev/null | head -20
echo
echo '=== 文件名里含 skinny / 4090 的 ==='
find /home/user /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -type f \( -iname '*skinny*' -o -iname '*4090*' \) 2>/dev/null | head -20
echo
echo '=== 内容里出现 v100-skinny / ninfer-4090 的文件 ==='
grep -rl --include='*.md' --include='*.sh' --include='*.py' --include='*.txt' --include='*.log' -e 'v100-skinny' -e 'v100_skinny' -e 'ninfer-4090' -e 'ninfer_4090' /mnt/c/Users/User/Documents/ziqinzhang 2>/dev/null | head -12
echo
echo '=== 内容里出现 skinny 的文件（限顶层文档）==='
grep -rl -i 'skinny' /mnt/c/Users/User/Documents/ziqinzhang/*.md /mnt/c/Users/User/Documents/ziqinzhang/_collab/*.md 2>/dev/null | head -10
