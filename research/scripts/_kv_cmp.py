#!/usr/bin/env python3
"""plain vs spec 的逐位置 KV 对照（用引擎自带 KV dump；门已放宽到 Prefill|Verify）。

每个 dump 文件：`kvsrc_<id>_L<layer>_{kn,v,pos}.bin` = 裸字节；`_meta.txt` 记 ne/nb/dtype。
对齐键 = (layer, 该位置在 pos 里的绝对位置)；比较 kn 与 v 的**逐字节**相等性。
判据：被接受前缀上的 (layer,pos) 必须逐位一致；首个不一致点即"写路径"疑点。
"""
import pathlib
import re
import struct
import sys

A_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvA")
B_DIR = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/kvB")
META = re.compile(r"^(kn|v|pos) ne=(\d+),(\d+),(\d+),(\d+) "
                  r"nb=(-?\d+),(-?\d+),(-?\d+),(-?\d+) dtype=(\d+)")


def read_group(directory):
    out = {}
    for meta in sorted(directory.glob("kvsrc_*_meta.txt")):
        m = re.match(r"kvsrc_(\d+)_L(\d+)_meta\.txt", meta.name)
        if not m:
            continue
        layer = int(m.group(2))
        info = {}
        for line in meta.read_text(errors="replace").splitlines():
            g = META.match(line.strip())
            if g:
                info[g.group(1)] = {"ne": [int(g.group(i)) for i in range(2, 6)],
                                    "nb": [int(g.group(i)) for i in range(6, 10)],
                                    "dtype": int(g.group(10))}
        if "kn" not in info or "v" not in info or "pos" not in info:
            continue
        kb = meta.with_name(meta.name.replace("_meta.txt", "_kn.bin")).read_bytes()
        vb = meta.with_name(meta.name.replace("_meta.txt", "_v.bin")).read_bytes()
        pb = meta.with_name(meta.name.replace("_meta.txt", "_pos.bin")).read_bytes()
        t = info["pos"]["ne"][0]
        pos = list(struct.unpack_from(f"<{t}i", pb, 0))
        # kn/v 的第 3 维 = token 索引；按 nb 求每 token 的字节块
        kn_row = info["kn"]["nb"][2] if info["kn"]["ne"][2] > 1 else info["kn"]["nb"][3]
        v_row = info["v"]["nb"][2] if info["v"]["ne"][2] > 1 else info["v"]["nb"][3]
        kn_row = kn_row if kn_row > 0 else len(kb) // max(t, 1)
        v_row = v_row if v_row > 0 else len(vb) // max(t, 1)
        for i, p in enumerate(pos):
            out[(layer, p)] = (kb[i * kn_row:(i + 1) * kn_row], vb[i * v_row:(i + 1) * v_row])
    return out


a = read_group(A_DIR)
b = read_group(B_DIR)
print(f"dump A 条目={len(a)}  dump B 条目={len(b)}")
common = sorted(set(a) & set(b))
print(f"公共 (layer,pos) 条目={len(common)}")
if not common:
    print("无可比条目：检查 dump 目录/层选择（NINFER_KVDUMP_KV）")
    raise SystemExit(0)

bad_k = [k for k in common if a[k][0] != b[k][0]]
bad_v = [k for k in common if a[k][1] != b[k][1]]
print()
print(f"  K 不一致: {len(bad_k)}/{len(common)}")
print(f"  V 不一致: {len(bad_v)}/{len(common)}")
if bad_k:
    k = bad_k[0]
    print(f"  首个 K 不一致: layer={k[0]} pos={k[1]}  "
          f"bytesA[0:8]={a[k][0][:8].hex()} bytesB[0:8]={b[k][0][:8].hex()}")
if bad_v:
    k = bad_v[0]
    print(f"  首个 V 不一致: layer={k[0]} pos={k[1]}  "
          f"bytesA[0:8]={a[k][0][:8].hex()} bytesB[0:8]={b[k][1][:8].hex()}")
print()
if not bad_k and not bad_v:
    print("⇒ 公共位置上的 K/V 全部逐位一致 ⇒ **写路径无误**，嫌疑转向读/注意力")
else:
    print("⇒ 存在 K/V 不一致 ⇒ **写路径疑点**（首个不一致点已给出）")
