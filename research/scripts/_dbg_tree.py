import pathlib
import re
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16


def load(p, f):
    raw = p.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}{f}", raw))


lg = (DL / "s3_scores.log").read_text(errors="replace")
pos_index, cur = {}, None
for line in lg.splitlines():
    if re.search(r"\[df2dbg\] row=(\d+)", line):
        cur = {}
        continue
    m = re.search(r"\[df2dbg\]\s+col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)", line)
    if m and cur is not None:
        col, pos, argmax = int(m.group(1)), int(m.group(2)), int(m.group(5))
        cur[col] = argmax
        if col == 0:
            pos_index[pos] = dict(cur)

for n in (6, 7, 8, 10, 12):
    fs = DL / f"df2scores_scores_{n}.bin"
    scores = load(fs, "f")
    cand = load(DL / f"df2scores_cand_{n}.bin", "i")
    front = load(DL / f"df2scores_front_{n}.bin", "i")
    steps = len(scores) // (K * K)
    tgt = pos_index.get(front[0])
    print(f"--- call {n}  frontier={front[0]}  steps={steps}")
    print(f"    tgt(按列) = {tgt}")
    # 深度 0 的候选与真值
    cl0 = [cand[0 + steps * c] for c in range(K)]
    print(f"    深度0候选 = {cl0}")
    cl1 = [cand[1 + steps * c] for c in range(K)]
    print(f"    深度1候选 = {cl1}")
    if tgt:
        print(f"    深度0真值{tgt.get(0)} 在候选第 {cl0.index(tgt[0]) if tgt.get(0) in cl0 else '不在'}")
        print(f"    深度1真值{tgt.get(1)} 在候选第 {cl1.index(tgt[1]) if tgt.get(1) in cl1 else '不在'}")
    # 贪心路径
    prev, path = 0, []
    for s in range(steps):
        bv, bc = float("-inf"), 0
        for c in range(K):
            v = scores[s + steps * (prev + K * c)]
            if v > bv:
                bv, bc = v, c
        path.append(bc)
        prev = bc
    print(f"    贪心 rank 路径 = {path}")
