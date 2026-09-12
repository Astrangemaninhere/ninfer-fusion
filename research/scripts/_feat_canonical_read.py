#!/usr/bin/env python3
"""features / projected / context dump 的**权威读法**。

关键：引擎 Tensor 的 nb[0] 是连续轴（src/core/tensor.cpp:51-58），
因此 [D=25600, W, B] 的 dump 在内存里是"按列连续"：
    v = np.fromfile(p, dtype="<f2").reshape(W*B, 25600)     # 行 = (w,b) 组合, 列 = D
tap i 段 = v[row, i*5120:(i+1)*5120]
`prepare_ragged_prefix` 只填 live 列（decode 时 live = 1+accepted），其余列整列零填充 —— 那是设计，不是缺料。
"""
import sys, pathlib
import numpy as np

D, HID, W, B = 25600, 5120, 8, 1

def load(path, dim):
    raw = np.fromfile(path, dtype="<f2").astype(np.float32)
    if raw.size % dim != 0:
        raise SystemExit("元素数 %d 不是 %d 的整数倍" % (raw.size, dim))
    return raw.reshape(raw.size // dim, dim)

def report(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print("  缺文件:", p); return
    raw = np.fromfile(p, dtype="<f2")
    nz = np.nonzero(raw)[0]
    span = (int(nz[0]), int(nz[-1]) + 1) if nz.size else (0, 0)
    print("  %s: 元素=%d 非零=%d 非零span=%s" % (p.name, raw.size, nz.size, span))
    if raw.size % D == 0 and raw.size // D <= 8:
        v = raw.astype(np.float32).reshape(raw.size // D, D)
        for r in range(min(v.shape[0], 4)):
            taps = [float(np.linalg.norm(v[r, i*HID:(i+1)*HID])) for i in range(5)]
            if any(taps):
                print("     列%d tap 范数 = %s" % (r, ["%.2f" % t for t in taps]))
    else:
        print("     （非 features 形状，跳过 tap 分段）")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        report(a)
