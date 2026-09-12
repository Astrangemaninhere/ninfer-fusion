#!/usr/bin/env python3
"""逐位置定位 chunk 形状分叉：同一批 token 在 chunk 128 vs 4096 下，逐位置比较
(a) 引擎自带的全词表逐列 argmax、(b) post-final-norm hidden、(c) 5 层目标特征。

prompt 取 ~900 token 且**非 128 对齐**（900 = 7*128 + 4），使两次运行的末块分别为 T=4 与 T=900，
既能触发 S_A ① 的双路（<64 走 FP32 寄存器），又让两者 chunk ≤1024（TOPK 记录可用）。
"""
import pathlib
import struct
import subprocess
import sys

BUILD = "/home/user/ninfer-fusion/build"
ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
SEG = "杭州地处长江三角洲南翼，浙江省北部，北靠天目山，南临钱塘江，水网密布、湖荡众多。"
FEATURE_ROWS = 25600
HIDDEN = 5120
MAGIC = 0x4E485331


def run(tag, chunk):
    out = pathlib.Path(f"/tmp/hs{tag}")
    subprocess.run(["rm", "-rf", str(out)], check=False)
    out.mkdir(parents=True, exist_ok=True)
    prompt = pathlib.Path(f"/tmp/hs_prompt_{tag}.txt")
    prompt.write_text(SEG * 30, encoding="utf-8")
    env = {"PATH": "/home/user/.local/bin:/usr/bin:/bin", "NINFER_HS_DUMP_DIR": str(out),
           "NINFER_HS_DUMP_TOPK": "1"}
    log = pathlib.Path(f"/tmp/hs_{tag}.log")
    with open(log, "wb") as handle:
        subprocess.run([f"{BUILD}/apps/ninfer", ART, "--prompt", prompt.read_text(encoding="utf-8"),
                        "--max-new", "8", "--max-context", "4096", "--no-thinking", "--greedy",
                        "--print-token-ids", "--spec", "dflash2",
                        "--prefill-chunk", str(chunk)],
                       env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=900)
    files = sorted(out.glob("chunk_*.bin"))
    print(f"  [{tag}] chunk={chunk} files={len(files)}")
    return files


def read_records(files):
    recs = []
    for path in files:
        raw = path.read_bytes()
        off = 0
        magic, = struct.unpack_from("<I", raw, off); off += 4
        if magic != MAGIC:
            raise SystemExit(f"bad magic in {path}")
        tokens, = struct.unpack_from("<i", raw, off); off += 4
        ids = list(struct.unpack_from(f"<{tokens}i", raw, off)); off += 4 * tokens
        feat = raw[off:off + FEATURE_ROWS * tokens * 2]; off += FEATURE_ROWS * tokens * 2
        last = raw[off:off + HIDDEN * tokens * 2]; off += HIDDEN * tokens * 2
        topk = None
        if off + 4 * tokens <= len(raw):
            topk = list(struct.unpack_from(f"<{tokens}i", raw, off))
        recs.append({"ids": ids, "feat": feat, "last": last, "topk": topk, "tokens": tokens})
    return recs


def flatten(recs, key):
    out = []
    for r in recs:
        if key == "topk":
            if r["topk"] is None:
                return None
            out.extend(r["topk"])
        elif key == "ids":
            out.extend(r["ids"])
        else:
            out.append(r[key])
    return out


def main() -> int:
    print("=== 运行两次（同一 prompt，只改 --prefill-chunk）===")
    fa = run("A128", 128)
    fb = run("B4096", 4096)
    ra, rb = read_records(fa), read_records(fb)
    ta = sum(r["tokens"] for r in ra)
    tb = sum(r["tokens"] for r in rb)
    print(f"  tokens: A={ta} B={tb}")
    ida, idb = flatten(ra, "ids"), flatten(rb, "ids")
    if ida != idb:
        print("  !! token id 序列不同（先确认 prompt 一致）")
        return 1
    topa, topb = flatten(ra, "topk"), flatten(rb, "topk")
    print()
    print("=== 逐位置比较（同 token id 对齐）===")
    if topa is None or topb is None:
        print(f"  TOPK 记录缺失 (A={'有' if topa else '无'}, B={'有' if topb else '无'})")
    else:
        n = min(len(topa), len(topb))
        bad = [i for i in range(n) if topa[i] != topb[i]]
        print(f"  引擎逐列 argmax：{len(bad)}/{n} 个位置不同"
              + (f"，首个 = 位置 {bad[0]}（A={topa[bad[0]]} B={topb[bad[0]]}）" if bad else "（全部相同）"))
        if bad:
            i = bad[0]
            lo = max(0, i - 3)
            print(f"    A[{lo}:{i+4}] = {topa[lo:i+4]}")
            print(f"    B[{lo}:{i+4}] = {topb[lo:i+4]}")
    # hidden 与 feature 按 bf16 字节比较（bf16 精度有限，仅作辅助）
    for key, label in (("last", "post-norm hidden (bf16)"), ("feat", "5 层特征 (bf16)")):
        va, vb = flatten(ra, key), flatten(rb, key)
        if va is None or vb is None or len(va) != len(vb):
            print(f"  {label}: 不可比")
            continue
        n = min(len(va), len(vb))
        first = next((k for k in range(n) if va[k] != vb[k]), None)
        cnt = sum(1 for k in range(n) if va[k] != vb[k])
        print(f"  {label}: {cnt}/{n} 个 chunk 字节不同"
              + (f"，首个 chunk = {first}" if first is not None else "（全部相同）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
