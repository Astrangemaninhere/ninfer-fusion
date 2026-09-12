#!/bin/bash
# 跑 MTP logits 分析并把输出写到 C: 侧可读文件
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
python3 /home/user/mtplan.py 3 > "$J/_mtp_analysis.txt" 2>&1
echo "written"
