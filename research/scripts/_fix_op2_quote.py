#!/usr/bin/env python3
"""修 _offset_probe2.sh 里的引号（Python 字符串里嵌了裸双引号）——一次性。"""
import pathlib

p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_offset_probe2.sh")
s = p.read_text(encoding="utf-8")
s = s.replace('固定在"第 36 个生成 token"这个计数上', '固定在「第 36 个生成 token」这个计数上')
p.write_text(s, encoding="utf-8")
print("ok; 检查是否还有裸双引号在中文句中:", '"第 36' in s)
