import pathlib, datetime

# 1) 修正 A5 报告里会误导的切法说明（只加澄清，不改原结论）
p5 = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_collab\A5_dspark_rootcause.md")
old = p5.read_text(encoding="utf-8", errors="replace")
note = """
---

## 附注（2026-09-12）：§1.3 那套 `feat[5L*5120:(5L+1)*5120]` 切法的适用边界

**该切法只在 prefill dump 上成立，在 decode dump 上会给出错误结论。**

原因：`features` 是 `[D=5*hidden=25600, width, batch]`，而引擎 `Tensor` 的 **`nb[0]` 才是连续轴**
（`src/core/tensor.cpp:51-58 set_contiguous_strides`），即 **D 是最快轴**。因此内存里的排布是
"按列（位置）连续"，一列 = 一个位置的 25600 个值（5 个 tap 各 5120）。而 decode 轮次里
`prepare_ragged_prefix`（`src/ops/kernel/prepare_ragged_prefix.cuh:17-26`，`live = column < count`，
`count = ends - starts`）**只填 live 列、其余列整列零填充**，且 decode 时
`count = 1 + accepted = 1`（接受率≈0 时）⇒ 非零内容只有第 0 列。

若按 row-major `(25600, W)` 读并沿行切 5 段，就会看到"第 0 段有值、后 4 段为零"，
从而误判为"只有 tap0 被写入"。**判别式**：正确几何下非零元素应为 `live*25600` 个连续值
（decode 时 live=1 ⇒ 恰好 25600 个、offset [0,25599]）；错误读法下"每列非零数 = D/W"（如 3200），
而若真为 tap0 独有则应为 5120。

**正确读法**：`v = np.fromfile(f, dtype='<f2').reshape(W*batch, 25600)`，
tap i 取 `v[row, i*5120:(i+1)*5120]`；decode 时 tap 范数示例（call5）
`[102.36, 118.58, 119.42, 127.74, 149.55]`（单调递增 = 层深递增 ⇒ 5 层都在）。
"""
p5.write_text(old + note, encoding="utf-8")
print("A5 报告已加附注, 现在", len(p5.read_text(encoding='utf-8', errors='replace').splitlines()), "行")

# 2) 写一个权威的读取脚本，防止同类误读再现
canon = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_feat_canonical_read.py")
canon.write_text('''#!/usr/bin/env python3
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
''', encoding="utf-8")
print("权威读法脚本已写出: _feat_canonical_read.py")
