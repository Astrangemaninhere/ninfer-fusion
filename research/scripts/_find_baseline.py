#!/usr/bin/env python3
"""搜 .ninfer 里 tokenizer.json 的真实起点，定标 offset 基准。"""
import os
P = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
size = os.path.getsize(P)
with open(P, "rb") as f:
    head = f.read(64 * 1024 * 1024)
print("文件大小 = %d" % size)
for pat in (b'{"version"', b'{"model"', b'{"added_tokens"', b'{"tokenizer_class"'):
    i = head.find(pat)
    print("  pat %-22r -> %s" % (pat, i))
i = head.find(b'{"version"')
print()
print("tokenizer.json 实际起点 = %s" % i)
print("manifest 里 offset = 0  => 基准 = %s" % i)
print("16+json_len = %d" % (16 + 176842))
print("=== 若 tokenizer.json 起点 = baseline + 0，则 baseline = %s ===" % i)
# 验证另一个资源：tokenizer_config.json 应在 tokenizer.json + 12809320
if i:
    j = head.find(b'{"', i + 12809320 - 32)
    print("在 baseline+12809320-32 附近找到的 JSON 起点 = %s （期望 ~%d）" % (j, i + 12809320))
