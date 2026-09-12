
# 长上下文质量对照 (needle-in-a-haystack, 2026-09-10 15:43)

| 配置 | ctx | needle hits | backend |
|---|---|---|---|
| baseline | 16384 | 5/8 | speculative=off |
| baseline | 32768 | 7/8 | speculative=off |
| mtp3 | 16384 | 6/8 | speculative=mtp |
| mtp3 | 32768 | 6/8 | speculative=mtp |
| mtp5 | 16384 | 4/8 | speculative=mtp |
| mtp5 | 32768 | 7/8 | speculative=mtp |

判读: MTP 的命中数必须与基线**相同**; 任一 ctx 下低于基线即质量回归
      (掉针 = 检索失败 = 输出不可信, 与速度无关)。
