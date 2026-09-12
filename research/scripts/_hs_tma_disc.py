#!/usr/bin/env python3
"""S_E 的单变量判别：prompt≈1024，`--prefill-chunk 128`（永不 TMA）vs `512`（恒 TMA）。
若两者 argmax 翻转率 ≈3~4%（与 128 vs 4096 同量级）⇒ TMA epilogue 缺 bf16 回舍 是主因；
若为 0 ⇒ TMA 清白，主因转向 gating proj 的 SplitK 分档。
"""
import pathlib
import struct
import subprocess
import sys

BUILD = "/home/user/ninfer-fusion/build"
ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
SEG = "杭州地处长江三角洲南翼，浙江省北部，北靠天目山，南临钱塘江，水网密布、湖荡众多。"
FEATURE_ROWS, HIDDEN, MAGIC = 25600, 5120, 0x4E485331


def run(tag, chunk, reps):
    out = pathlib.Path(f"/tmp/hs{tag}")
    subprocess.run(["rm", "-rf", str(out)], check=False)
    out.mkdir(parents=True, exist_ok=True)
    prompt = SEG * reps
    env = {"PATH": "/home/user/.local/bin:/usr/bin:/bin", "NINFER_HS_DUMP_DIR": str(out),
           "NINFER_HS_DUMP_TOPK": "1"}
    log = pathlib.Path(f"/tmp/hs_{tag}.log")
    with open(log, "wb") as handle:
        subprocess.run([f"{BUILD}/apps/ninfer", ART, "--prompt", prompt, "--max-new", "4",
                        "--max-context", "4096", "--no-thinking", "--greedy",
                        "--print-token-ids", "--spec", "dflash2", "--prefill-chunk", str(chunk)],
                       env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=900)
    files = sorted(out.glob("chunk_*.bin"))
    print(f"  [{tag}] chunk={chunk} reps={reps} files={len(files)}")
    return files


def parse(files):
    rows, pos = {}, 0
    for path in files:
        raw = path.read_bytes()
        off = 0
        magic, = struct.unpack_from("<I", raw, off); off += 4
        assert magic == MAGIC, path
        tokens, = struct.unpack_from("<i", raw, off); off += 4
        ids = list(struct.unpack_from(f"<{tokens}i", raw, off)); off += 4 * tokens
        off += FEATURE_ROWS * tokens * 2
        last = raw[off:off + HIDDEN * tokens * 2]; off += HIDDEN * tokens * 2
        topk = None
        if off + 4 * tokens <= len(raw):
            topk = list(struct.unpack_from(f"<{tokens}i", raw, off))
        for t in range(tokens):
            rows[pos + t] = {"id": ids[t],
                             "last": last[t * HIDDEN * 2:(t + 1) * HIDDEN * 2],
                             "topk": None if topk is None else topk[t]}
        pos += tokens
    return rows


def main() -> int:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 34
    a = parse(run("T128", 128, reps))
    b = parse(run("T512", 512, reps))
    common = sorted(set(a) & set(b))
    print(f"  位置数 A={len(a)} B={len(b)} 公共={len(common)}")
    if not common or any(a[p]["id"] != b[p]["id"] for p in common):
        print("!! id 不一致"); return 1
    dn = sum(1 for p in common if a[p]["last"] != b[p]["last"])
    dd = [p for p in common if a[p]["last"] != b[p]["last"]]
    tk = 0
    tkl = []
    for p in common:
        if a[p]["topk"] is not None and b[p]["topk"] is not None and a[p]["topk"] != b[p]["topk"]:
            tk += 1
            tkl.append(p)
    n = len(common)
    print()
    print(f"  post-norm hidden 不同: {dn}/{n}")
    print(f"  引擎逐列 argmax 不同 : {tk}/{n} = {100.0*tk/n:.2f}%"
          + (f"  首个位置={tkl[0]}" if tkl else "（全部相同）"))
    print(f"  首次 hidden 差异位置 : {dd[0] if dd else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
