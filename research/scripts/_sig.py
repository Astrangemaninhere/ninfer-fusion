#!/usr/bin/env python3
"""深度签名（与参考同口径）：
  P(top1[d]==top1[d+1]) / 游程直方图 / 最大游程 / 相邻深度 top-16 的 Jaccard /
  以及"top-1 重复时 vs 不同时"的 Jaccard 均值（参考的耦合特征：重复⇒集合高度重叠）

参考侧实测（agent 在 vLLM 上 dump 的 21 个真实步 × 7 深度 = 147 槽）：
  P(相邻 top1 相同) = 9.5%（12/126）；游程直方图 {1:9, 2:12}；**无 >=3 的游程**
  相邻 Jaccard 均值 0.303；重复时 Jaccard 均值 0.449 vs 不同时 0.287
"""
import pathlib
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16


def load_i32(p):
    raw = p.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}i", raw))


def main():
    trans_same = trans_tot = 0
    runs = []
    jac_same, jac_diff = [], []
    maxrun = 0
    for n in range(1, 40):
        fs = DL / f"df2scores_scores_{n}.bin"
        if not fs.exists():
            break
        cand = load_i32(DL / f"df2scores_cand_{n}.bin")
        steps = len(fs.read_bytes()) // 4 // (K * K)

        def cd(s, c):
            return cand[s + steps * c]

        tops = [cd(s, 0) for s in range(steps)]
        sets = [set(cd(s, c) for c in range(K)) for s in range(steps)]
        # 游程
        cur, rr = 1, []
        for s in range(1, steps):
            if tops[s] == tops[s - 1]:
                cur += 1
            else:
                rr.append(cur)
                cur = 1
        rr.append(cur)
        runs.extend(rr)
        maxrun = max(maxrun, max(rr))
        for s in range(steps - 1):
            same = tops[s] == tops[s + 1]
            trans_tot += 1
            trans_same += same
            inter = len(sets[s] & sets[s + 1])
            uni = len(sets[s] | sets[s + 1])
            j = inter / uni if uni else 0.0
            (jac_same if same else jac_diff).append(j)

    hist = {}
    for r in runs:
        hist[r] = hist.get(r, 0) + 1
    print(f"我们的深度签名（{len(runs)} 段游程，共 {trans_tot} 次相邻转移）:")
    print(f"  P(top1[d]==top1[d+1]) = {trans_same}/{trans_tot} = {100.0*trans_same/max(trans_tot,1):.1f}%")
    print(f"  游程直方图 = {dict(sorted(hist.items()))}")
    print(f"  最大游程 = {maxrun}   （参考: 2）")
    print(f"  相邻 Jaccard 均值 = {sum(jac_same+jac_diff)/max(len(jac_same+jac_diff),1):.3f}  （参考: 0.303）")
    ms = sum(jac_same) / len(jac_same) if jac_same else float("nan")
    md = sum(jac_diff) / len(jac_diff) if jac_diff else float("nan")
    print(f"  重复时 Jaccard 均值 = {ms:.3f}（参考 0.449） / 不同时 = {md:.3f}（参考 0.287）")
    print(f"  ⇒ 耦合方向：{'与参考一致(重复⇒集合重叠)' if ms > md else '与参考相反(重复时集合仍在变)'}")


if __name__ == "__main__":
    main()
