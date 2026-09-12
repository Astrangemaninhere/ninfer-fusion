#!/usr/bin/env python3
"""核心注意力因果性判别（零改码）。

问题：target 的 verify 块里，第 0 列（输入=真 anchor、上下文=真实已接受前缀）的 argmax
      会不会被**同一块右侧的 draft 列**影响？若会 ⇒ 掩码非因果 ⇒ 与形状无关的真 bug。
方法：同一 prompt 跑两次，只用 `--lm-head-draft` 换一组**不同的草稿**（其余全同）；
      对每个 run 还原"每轮的上下文前缀 + 该轮第 0 列 argmax"，把两轮按
      **相同上下文前缀**配对（同前缀、同位置、同输入 token）⇒ 比较第 0 列 argmax。
      因果 ⇒ 第 0 列 argmax 必须逐位相同；非因果 ⇒ 会出现不同。
控制：同时打印配对轮里两 run 的右侧 draft 列，确认它们**确实不同**（否则实验无力度）。
"""
import pathlib
import re
import subprocess
import sys

BUILD = "/home/user/ninfer-fusion/build"
ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
PROMPT = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"
ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")


def run(tag, extra):
    log = pathlib.Path(f"/tmp/caus_{tag}.log")
    env = {"PATH": "/home/user/.local/bin:/usr/bin:/bin", "NINFER_DF2DBG": "1"}
    with open(log, "wb") as handle:
        subprocess.run([f"{BUILD}/apps/ninfer", ART, "--prompt", PROMPT, "--max-new", "96",
                        "--max-context", "4096", "--no-thinking", "--greedy", "--spec", "dflash2"]
                       + extra, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=900)
    print(f"  [{tag}] {extra or '(full head)'} log={log}")
    return log


def parse(path):
    rounds, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m:
            cur = {"accepted": int(m.group(7)), "cols": []}
            rounds.append(cur)
            continue
        m = COL.search(line)
        if m and cur is not None:
            cur["cols"].append({"col": int(m.group(1)), "pos": int(m.group(2)),
                                "verify": int(m.group(3)), "draft": int(m.group(4)),
                                "argmax": int(m.group(5))})
    return rounds


def trajectory(rounds):
    """还原 (上下文前缀 tuple, 该轮 col0 argmax, 右侧 drafts) 序列。滞后校正：own=下一行 accepted。"""
    out, prefix = [], []
    for i, r in enumerate(rounds):
        own = rounds[i + 1]["accepted"] if i + 1 < len(rounds) else None
        if own is None or not r["cols"]:
            continue
        a = max(0, own)
        drafts = [c["draft"] for c in r["cols"] if c["draft"] != -1]
        out.append({"prefix": tuple(prefix), "pos": r["cols"][0]["pos"],
                    "anchor": r["cols"][0]["verify"], "col0": r["cols"][0]["argmax"],
                    "rest": drafts[1:] if len(drafts) > 1 else []})
        prefix.extend(drafts[:a])
        if r["cols"]:
            prefix.append(r["cols"][min(a, len(r["cols"]) - 1)]["argmax"])
    return out


def main() -> int:
    print("=== 两次运行（仅换草稿来源） ===")
    a = trajectory(parse(run("full", [])))
    b = trajectory(parse(run("lmhd", ["--lm-head-draft"])))
    print(f"  轮数: A={len(a)} B={len(b)}")

    idx = {}
    for r in b:
        idx.setdefault((r["prefix"], r["pos"]), []).append(r)
    pairs = []
    for r in a:
        k = (r["prefix"], r["pos"])
        if k in idx:
            for q in idx[k]:
                pairs.append((r, q))
    print(f"  可配对的轮（同上下文前缀 + 同位置）: {len(pairs)}")
    if not pairs:
        print("  => 无配对：两次运行的路径未在任何轮重合，需改用更短 prompt 或更多轮")
        return 0
    diff = [p for p in pairs if p[0]["col0"] != p[1]["col0"]]
    print()
    print(f"  第 0 列 argmax 不同者: {len(diff)}/{len(pairs)}  "
          f"({100.0*len(diff)/len(pairs):.1f}%)")
    print(f"  右侧 draft 列不同者  : "
          f"{sum(1 for p in pairs if p[0]['rest'] != p[1]['rest'])}/{len(pairs)}  ← 控制项（应有差异）")
    print()
    for r, q in pairs[:6]:
        print(f"  pos={r['pos']} anchor={r['anchor']} prefix_len={len(r['prefix'])}")
        print(f"     A col0={r['col0']}  rest={r['rest'][:4]}")
        print(f"     B col0={q['col0']}  rest={q['rest'][:4]}"
              + ("   <-- col0 不同（非因果？）" if r["col0"] != q["col0"] else ""))
    print()
    if diff:
        print("⇒ 存在 col0 不同 ⇒ **target verify 的第 0 列受右侧 draft 列影响（掩码非因果）**")
    else:
        print("⇒ 配对轮里 col0 全部相同 ⇒ 因果性成立（核心掩码不是本因）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
