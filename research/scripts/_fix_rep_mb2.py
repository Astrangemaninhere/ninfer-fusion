#!/usr/bin/env python3
"""按实际行文本修体积公式与文案（上一版锚点没对上，未写入）。"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)
touched = 0
for i, ln in enumerate(lines):
    if ln.strip().startswith("rep_mb = (g0.get('query_heads', 0) * g0.get('hidden', 0) * 2 * 3)"):
        lines[i] = ("        # 让元素级 sigmoid_mul 等价于逐头广播，需要把 g_proj 展开成\n"
                    "        # [n_q*head_dim, hidden] 并把行按 head 复制 head_dim 次；\n"
                    "        # BF16 体积 = n_q*head_dim*hidden*2*层数（Spark: 16*256*2560*2*36 ≈ 755 MB）。\n"
                    "        rep_mb = (g0.get('query_heads', 0) * g0.get('head_dim', 0) * g0.get('hidden', 0)\n"
                    "                  * 2 * g0.get('layers', 0)) / 1e6\n")
        touched += 1
    elif "零算子出路=复制 g_proj 权重" in ln:
        lines[i] = ("                    '逐头广播无算子; 零算子出路=把 g_proj 展开成 [n_q*head_dim, hidden] '\n"
                    "                    '的元素级门控(约 +%.0f MB 权重) 或新内核' % rep_mb))\n")
        touched += 1
assert touched == 2, "只改了 %d 处，应为 2" % touched
P.write_text("".join(lines), encoding="utf-8")
print("已修 2 处")
