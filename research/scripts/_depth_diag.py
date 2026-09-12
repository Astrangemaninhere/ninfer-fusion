#!/usr/bin/env python3
"""深度结构诊断：定位"深度>=2 接受精确为零"。

看点：
 1) 逐深度的 unary top-1 token —— 深度 1..6 是否提出同一个 token（MASK 退化的标志）。
 2) 相邻深度的 top-16 集合重合度（Jaccard）。
 3) proposal_hidden / proj 的逐深度 RMS —— 深度 >=1 是否量级异常。
 4) 头 logits 的逐深度 RMS。
"""
import pathlib
import struct

DL = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
K = 16
HID = 5120
RANK = 256
VOCAB = 248320


def load_i32(p):
    raw = p.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}i", raw))


def load_bf16_to_float(p):
    raw = p.read_bytes()
    import numpy as np
    a = np.frombuffer(raw, dtype="<u2").astype("uint32") << 16
    return a.view("float32")


def main():
    for n in (1, 3, 6, 10):
        fs = DL / f"df2scores_scores_{n}.bin"
        if not fs.exists():
            continue
        cand = load_i32(DL / f"df2scores_cand_{n}.bin")
        scores_raw = fs.read_bytes()
        steps = len(scores_raw) // 4 // (K * K)

        ph = load_bf16_to_float(DL / f"df2scores_ph_{n}.bin")     # [HID, steps] 行主序
        proj = load_bf16_to_float(DL / f"df2scores_proj_{n}.bin")  # 实为 FP32，这里仅取字节统计
        lg = load_bf16_to_float(DL / f"df2scores_logits_{n}.bin")  # [VOCAB, steps]

        def cd(s, c):
            return cand[s + steps * c]

        tops = [cd(s, 0) for s in range(steps)]
        sets = [set(cd(s, c) for c in range(K)) for s in range(steps)]
        print(f"\n===== call {n}  steps={steps} =====")
        print(f"  逐深度 unary top-1 = {tops}")
        print(f"  深度 1..{steps-1} 是否全同 = {len(set(tops[1:])) == 1}")
        jac = []
        for s in range(steps - 1):
            inter = len(sets[s] & sets[s + 1])
            uni = len(sets[s] | sets[s + 1])
            jac.append(inter / uni if uni else 0)
        print(f"  相邻深度 top-16 的 Jaccard = {[round(x,2) for x in jac]}")
        if ph.size == HID * steps:
            rms = [float((ph[s::steps] ** 2).mean() ** 0.5) for s in range(steps)]
            print(f"  proposal_hidden 逐深度 RMS = {[round(x,2) for x in rms]}")
        if lg.size == VOCAB * steps:
            rms = [float((lg[s::steps] ** 2).mean() ** 0.5) for s in range(steps)]
            mx = [float(lg[s::steps].max()) for s in range(steps)]
            nmax = [int((lg[s::steps] == lg[s::steps].max()).argmax()) for s in range(steps)]
            print(f"  头 logits 逐深度 RMS = {[round(x,2) for x in rms]}")
            print(f"  头 logits 逐深度 max token id = {nmax}")


if __name__ == "__main__":
    main()
