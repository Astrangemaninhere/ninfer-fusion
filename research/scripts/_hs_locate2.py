#!/usr/bin/env python3
"""按 token 位置 + 逐层定位：chunk 128 vs 4096 的 dump 对比（复用已产出的 /tmp/hs{A128,B4096}）。

每个 chunk 记录：magic, tokens, ids[], feat[25600 x tokens](bf16), last[5120 x tokens](bf16), topk[tokens]。
特征 25600 = 5 层 × 5120 ⇒ 可判**哪一层**先在哪个位置分叉。
"""
import pathlib
import struct
import sys

FEATURE_ROWS, HIDDEN, MAGIC = 25600, 5120, 0x4E485331
LAYERS = [(5, 25600, 0), (19, 25600, 10240), (33, 25600, 20480),
          (47, 25600, 30720), (61, 25600, 40960)]     # (层号, 行长, 字节偏移)


def parse(files):
    rows, pos = {}, 0
    for path in files:
        raw = path.read_bytes()
        off = 0
        magic, = struct.unpack_from("<I", raw, off); off += 4
        assert magic == MAGIC, path
        tokens, = struct.unpack_from("<i", raw, off); off += 4
        ids = list(struct.unpack_from(f"<{tokens}i", raw, off)); off += 4 * tokens
        feat = raw[off:off + FEATURE_ROWS * tokens * 2]; off += FEATURE_ROWS * tokens * 2
        last = raw[off:off + HIDDEN * tokens * 2]; off += HIDDEN * tokens * 2
        topk = None
        if off + 4 * tokens <= len(raw):
            topk = list(struct.unpack_from(f"<{tokens}i", raw, off))
        for t in range(tokens):
            rows[pos + t] = {
                "id": ids[t],
                "feat": feat[t * FEATURE_ROWS * 2:(t + 1) * FEATURE_ROWS * 2],
                "last": last[t * HIDDEN * 2:(t + 1) * HIDDEN * 2],
                "topk": None if topk is None else topk[t],
            }
        pos += tokens
    return rows


def main() -> int:
    A = parse(sorted(pathlib.Path("/tmp/hsA128").glob("chunk_*.bin")))
    B = parse(sorted(pathlib.Path("/tmp/hsB4096").glob("chunk_*.bin")))
    print(f"位置数: A={len(A)} B={len(B)}")
    common = sorted(set(A) & set(B))
    if not common:
        print("无公共位置"); return 1
    if any(A[p]["id"] != B[p]["id"] for p in common):
        print("!! token id 序列不同")
        return 1

    first = {"feat": None, "last": None, "topk": None}
    cnt = {"feat": 0, "last": 0, "topk": 0}
    layer_first = {}
    for p in common:
        a, b = A[p], B[p]
        if a["last"] != b["last"]:
            cnt["last"] += 1
            first["last"] = first["last"] if first["last"] is not None else p
        if a["feat"] != b["feat"]:
            cnt["feat"] += 1
            first["feat"] = first["feat"] if first["feat"] is not None else p
            for (layer, _rowlen, byte_off) in LAYERS:
                if a["feat"][byte_off:byte_off + HIDDEN * 2] != b["feat"][byte_off:byte_off + HIDDEN * 2]:
                    layer_first.setdefault(layer, p)
        if a["topk"] is not None and b["topk"] is not None and a["topk"] != b["topk"]:
            cnt["topk"] += 1
            first["topk"] = first["topk"] if first["topk"] is not None else p

    n = len(common)
    print()
    print(f"{'项':<34}{'不同位置数':>10}{'首次位置':>10}")
    print(f"{'post-norm hidden (bf16, 5120)':<34}{cnt['last']:>10}{str(first['last']):>10}")
    print(f"{'5 层特征任一不同 (bf16, 25600)':<34}{cnt['feat']:>10}{str(first['feat']):>10}")
    print(f"{'引擎逐列 argmax (全词表)':<34}{cnt['topk']:>10}{str(first['topk']):>10}")
    print(f"（共 {n} 个位置）")
    print()
    if layer_first:
        print("按捕获层看**首次**出现差异的位置（层号 → 首次位置）:")
        for layer in sorted(layer_first):
            print(f"    layer {layer:<4} → pos {layer_first[layer]}")
    else:
        print("5 层特征全部逐位相同（差异出现在更晚的层或 final_norm/head）")
    p = first["feat"] if first["feat"] is not None else first["last"]
    if p is not None:
        lo = max(0, p - 3)
        print()
        print(f"首个差异位置附近 pos {lo}..{p+3}:  id A={[A[q]['id'] for q in range(lo, min(p+4, max(common)+1)) if q in A]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
