#!/usr/bin/env python3
"""用真实 cache 自证 ids16 的口径：ids16[t,0] 是 tokens[t] 还是 tokens[t+1]？
这决定 `--target-shift` 该是 0 还是 1（即 dflash2 的训练目标是否整体偏移一行）。
同时给出"位置 a+i+1 的 token"对应的 teacher 行到底是哪一行。"""
import glob

import numpy as np

CACHE = r"C:\Users\User\Documents\ziqinzhang\data\hs_cache_topk2"
files = sorted(glob.glob(CACHE + r"\seq_*.npz"))[:4]
print("检查 %d 个 cache 文件" % len(files))
tot_next = tot_self = tot = 0
for p in files:
    z = np.load(p)
    t = z["tokens"].astype(np.int64)
    if "ids16" not in z.files:
        print("  %s: 无 ids16" % p.rsplit("\\", 1)[-1]); continue
    ids = z["ids16"].astype(np.int64)
    n = min(len(t) - 1, len(ids))
    if n <= 8:
        print("  %s: 太短 (%d)" % (p.rsplit("\\", 1)[-1], n)); continue
    nx = float((ids[:n, 0] == t[1:n + 1]).mean())   # ids16[t] 预测 tokens[t+1]
    sf = float((ids[:n, 0] == t[:n]).mean())        # ids16[t] 就是 tokens[t]
    tot_next += nx * n; tot_self += sf * n; tot += n
    print("  %-22s n=%5d  P(ids16[t]==tok[t+1])=%.4f  P(ids16[t]==tok[t])=%.4f"
          % (p.rsplit("\\", 1)[-1], n, nx, sf))
print()
print("汇总：P(next)=%.4f  P(self)=%.4f  （样本 %d）" % (tot_next / tot, tot_self / tot, tot))
print()
if tot_next > tot_self + 0.05:
    print("⇒ ids16[t] 是【位置 t 的 next-token 分布】(预测 tokens[t+1])")
    print("⇒ 那么『位置 p 的 token』的老师行是 ids16[p-1]；")
    print("  train_dflash2 的 block 槽位 i 对应位置 a+1+i，其目标行应为 ids16[a+i] ⇒ **--target-shift 应为 0**；")
    print("  默认 shift=1 取 ids16[a+1+i] = 位置 a+2+i 的 token ⇒ **训练目标整体晚一行**（用户从第一天就怀疑的那件事）。")
elif tot_self > tot_next + 0.05:
    print("⇒ ids16[t] 就是【tokens[t] 自己的分布】⇒ 『位置 p 的 token』对应行 ids16[p] ⇒ shift=1（默认）是对的。")
else:
    print("⇒ 口径不明确（两者接近），需要看采集脚本的 dump 代码。")
