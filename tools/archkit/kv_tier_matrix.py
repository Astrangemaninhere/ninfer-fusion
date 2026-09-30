#!/usr/bin/env python3
"""kv_tier_matrix.py — 自动化 KV 档位 × 层 标定流水线。

目标：为「哪一层用哪种 KV 量化」提供**可量化比较**的依据，判据同时含精度与速度，
且全部由实测产生、可自动复跑（不是一次性实验）。

测什么（全档位，自动展开）：
  A) 统一档位臂：16 层同档 ∈ {bf16, int8, fp8, nvfp4, iso4e, rk4v4}
     → 隔离「每档反量化/读取成本」。注意实际路径（勿凭档名推断）：
        - int8 活跃内核**不做旋转**（单趟 mma_s8 + 每 64 元素一个 fp16 scale）；
        - nvfp4 = E2M1 原生 QK（但**无条件跑第二趟残差 QK**）+ SO(4) 旋转 + Sinkhorn row scale
          + V 走软件解码(ISO4E) + BF16 PV；
        - iso4e 是独立内核：Iso4eGroup16 映射到 DType::ISO3，走
          gqa_attention_decode_iso3.cuh（K 面是 ISO4E 符号幅值 nibble，**不是 E2M1**）；
          与 nvfp4 只共享平面几何（两 code/byte + per-16 E4M3FN scale ⇒ 4.50 bits/el）、
          不共享内核，也不读 nvfp4 的第二级残差平面（decoder_state.cpp:174-196）。
          （旧注释写「映射到 DType::NVFP4 / 同内核」，已过时。）
        - rk4v4 = int8 内核 + Rk4v4 格点投影/H64 + 每 64 元素 scale。
  B) 单层差分臂：基线档 nvfp4，仅把第 L 层换成 rk4v4（L = 0..15）
     → 每层对「换便宜档」的**速度收益**与**精度代价**的归因（可加性稍后用双层臂校验）
  C) 可加性校验：同时换两层（首/中/尾各一组）+ `stack_factory_rk4v4`（出厂 6 层一起换，
     即 |D − Σd| 里的 D）。S 取自出厂表，不在这里重声明一份，免得与 variant.cpp 分叉。
  D) 默认臂：出厂表本身作为对照。
     ⚠️ 那份表今天是**工作树**的 `{0,1,3,4,6,7}`（6 层），而 HEAD 是
     `{0,1,3,4,6,7,8,9,13,14}`（10 层）。两者都必须写明读的是哪一个，见
     src/targets/qwen3_6_27b/impl/variant.cpp 的 default_layer_kv_dtypes()。
  E) 重复度臂（新增）：`default_mixed` 与 `only_layer0_rk4v4` 各重跑 `--repeats` 次，
     由重复臂之间的差给出 **noise_floor σ**。没有 σ，C 组的 `|D − Σd|` 判据没有分母，
     不可证伪；所以 σ 是产出而不是假设。

每个臂的读数：decode tok/s、AL、graded 答案（长上下文针尖精确前缀长度 0..N）、
KV payload、审计行里解析出的档位混合、**该臂的 (K,V) 二元组**、以及逐字 flag 向量。
每个臂的身份还应带上 {binary sha256, artifact sha256, messages sha256, needle 逐字}，
由顶层的 `provenance` 块给出；语料的 unique_lines / max_line_repeat / degenerate 由
`--corpus` 给出 —— 只做 sha256 的守卫对「同一行重复 120 次」完全盲。

输出：JSON（机器可读，供分配器/DP 消费）+ 排序表 + 首版建议；
加 `--w-matrix-out` 时另出一份 W[layer][format]，每格自带 provenance 与
`measured` 标志，可直接喂给 src/product/kv_adapt_solver.h 的 KvAdaptRequest。

用法（在本机，长跑务必后台）：
  python3 tools/archkit/kv_tier_matrix.py --out /path/table.json \
      --w-matrix-out /path/w.json --corpus /path/corpus.txt [--quick]

⚠️ 本脚本第 70 行起会执行 `pkill -9 -x ninfer`（`run_arm` 开头）。它不是无害的：
在还有别的线正在跑 ninfer 的机器上运行它，会**杀掉别人的测量**。落地的作序必须把
它排在「独占 GPU」的窗口里，而不是「我拿到了锁」的窗口里。
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Repository root derived from this file (tools/archkit/<file>): the default binary and
# message fixture follow the checkout instead of a machine path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# The artifact under test is an operator input; main() refuses to run without it.
MODEL = os.environ.get("KVX_MODEL")
BIN = os.environ.get("KVX_BIN", str(_REPO_ROOT / "build" / "apps" / "ninfer"))
MSGS = os.environ.get(
    "KVX_MESSAGES",
    str(_REPO_ROOT / "examples" / "cli" / "messages" / "long_niah_64k.json"))
NEEDLE = os.environ.get("KVX_NEEDLE", "ORCHID=493817; COLOR=COBALT")
TIERS = ["bf16", "int8", "fp8", "nvfp4", "iso4e", "rk4v4"]
LAYERS = 16

# ---------------------------------------------------------------------------
# (K, V) pairs, NOT tier names. A tier name is a label; the ablation's identity
# is the two planes it resolves to. This matters because the two are NOT
# interchangeable: nvfp4's V plane is ISO4E (Sinkhorn row scale + software V
# decode), so an ablation that swaps only the K plane measures a DIFFERENT
# thing from the deployed nvfp4 layer -- the KV-PERLAYER line measured that
# divergence at 4.24x-6.56x on NMSE. Recording the pair makes the mixup
# unrepresentable instead of merely discouraged.
FORMAT_KV = {
    "bf16":  ("bf16", "bf16"),
    "int8":  ("int8", "int8"),
    "fp8":   ("fp8", "fp8"),
    "nvfp4": ("nvfp4-e2m1", "iso4e"),
    "iso4e":  ("iso4e", "iso4e"),
    "rk4v4":    ("e8-lattice", "e8-lattice"),
    "cold":  ("rans-slot", "rans-slot"),
}


def sha256_file(path):
    """sha256 of a file, or None when it cannot be read (never a fabricated value)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def corpus_stats(path, mode):
    """unique_lines / max_line_repeat for the corpus, so a DEGENERATE corpus is visible.

    The guard today is a per-stream sha256 only, which is blind to the failure mode
    the recorded perplexity numbers actually have: the corpus is one 225-codepoint
    Chinese paragraph appended to itself 120 times, i.e. unique_lines == 1. A
    digest cannot see that; a line histogram can.
    """
    stats = {"mode": mode, "path": path, "sha256": None, "lines": None,
             "unique_lines": None, "max_line_repeat": None, "degenerate": None}
    if not path or not os.path.isfile(path):
        return stats
    stats["sha256"] = sha256_file(path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in (l.rstrip("\n") for l in fh)]
    except OSError:
        return stats
    nonempty = [ln for ln in lines if ln.strip()]
    stats["lines"] = len(lines)
    stats["unique_lines"] = len(set(nonempty))
    stats["max_line_repeat"] = (
        max((nonempty.count(ln) for ln in set(nonempty)), default=0))
    # A corpus whose distinct lines are fewer than a tenth of its lines cannot
    # exercise long-range recall; say so rather than let a reader assume it did.
    stats["degenerate"] = bool(nonempty) and stats["unique_lines"] * 10 < len(nonempty)
    return stats


def provenance_block(model, messages, needle):
    return {
        "binary_path": BIN,
        "binary_sha256": sha256_file(BIN),
        "artifact_path": model,
        "artifact_sha256": sha256_file(model),
        "artifact_size": (os.path.getsize(model) if model and os.path.isfile(model) else None),
        "messages_path": messages,
        "messages_sha256": sha256_file(messages),
        "needle_verbatim": needle,
    }



def spec_with(base, overrides):
    """base 为全局默认档，overrides = {layer: tier}；输出不重叠的连续区间语法。"""
    tiers = {}
    for l in range(LAYERS):
        tiers[l] = overrides.get(l, base)
    parts, start = [], 0
    for l in range(1, LAYERS + 1):
        if l == LAYERS or tiers[l] != tiers[start]:
            lo, hi = start, l - 1
            parts.append(f"{lo}:{tiers[start]}" if lo == hi else f"{lo}-{hi}:{tiers[start]}")
            start = l
    return ",".join(parts)

def run_arm(label, extra, max_new=32, timeout=1800):
    """跑一个臂，返回读数 dict。长上下文 + 贪心 ⇒ 确定性，单次即可。"""
    subprocess.run(["pkill", "-9", "-x", "ninfer"], check=False)
    time.sleep(3)
    argv = [BIN, MODEL, "--messages", MSGS, "--max-new", str(max_new),
            "--max-context", "131072", "--kv-capacity", "auto", "--no-thinking",
            "--greedy", "--spec", "mtp", "--draft-tokens", "9"] + list(extra)
    t0 = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 124, "", "timeout"
    wall = time.time() - t0

    def metric(pattern, group=1):
        m = re.search(pattern, err)
        return m.group(group) if m else None

    # graded：长上下文针尖答案的精确前缀长度（0..len(NEEDLE)）
    flat = re.sub(r"\s+", " ", out)
    graded = 0
    while graded < len(NEEDLE) and graded < len(flat) and flat[graded] == NEEDLE[graded]:
        graded += 1
    mix = metric(r"\[kv-bit-budget\][^\n]*") if "--kv-bit-budget" in " ".join(extra) else None
    return {
        "label": label, "extra": list(extra), "rc": rc, "wall_s": round(wall, 1),
        # The (K, V) pair this arm's ablation variable denotes, not the tier label.
        "kv_pair": [FORMAT_KV[t] for t in _tiers_in(extra)],
        "decode_tok_s": metric(r"decode speed\s+([\d.]+)"),
        "prefill_tok_s": metric(r"prefill speed\s+([\d.]+)"),
        "al": metric(r"acceptance length\s+([\d.]+)"),
        "kv_payload": metric(r"kv cache payload\s+([\d.]+ \w+)"),
        "kv_capacity": metric(r"KV capacity\s+(\d+)"),
        "graded_hit": graded, "graded_max": len(NEEDLE),
        "needle_ok": graded == len(NEEDLE),
        "hot": (mix or "").split("hot=")[-1].split(" (")[0] if mix else None,
        "error": metric(r"^error: (.+)$"),
    }


def _tiers_in(extra):
    """Tier names actually named by this arm's flags, in first-appearance order."""
    found = []
    for token in extra:
        for tier in TIERS:
            if re.search(rf"(^|[:\-,]){tier}\b", str(token)) and tier not in found:
                found.append(tier)
    return found


def noise_floor_of(arm_a, arm_b, key="graded_hit"):
    """The noise floor is a MEASUREMENT: the gap between two runs of the SAME arm.

    It cannot be assumed to be zero, and it cannot be inferred from a different
    arm. Additivity is checked against |D - sum(d_l)| <= sigma, so without this
    number that check has no denominator and the design's T4 is unfalsifiable.
    """
    va, vb = arm_a.get(key), arm_b.get(key)
    if va is None or vb is None:
        return None
    return abs(va - vb)


def build_w_matrix(arms, repeats, noise):
    """W[layer][format] from the B group, each cell carrying its own provenance.

    A cell with no repeat pair, or whose repeat pair disagrees by more than the
    noise floor, is emitted with measured=false. It is NEVER emitted as 0.0 --
    "nobody looked" and "the error is zero" must not be the same value, which is
    exactly the convention product/kv_perlayer_policy.h:85 already sets.
    """
    cells = []
    by_label = {a["label"]: a for a in arms}
    for arm in arms:
        m = re.match(r"only_layer(\d+)_(.+)$", arm["label"])
        if not m:
            continue
        layer, fmt = int(m.group(1)), m.group(2)
        rep = by_label.get(f"{arm['label']}_rep2")
        delta = None
        measured = False
        if rep is not None and rep.get("graded_hit") is not None:
            delta = abs((arm.get("graded_hit") or 0) - (rep.get("graded_hit") or 0))
            measured = noise is None or delta <= noise
        cells.append({
            "layer": layer,
            "format": fmt,
            "kv_pair": list(FORMAT_KV.get(fmt, ("", ""))),
            "loss": arm.get("graded_hit"),
            "loss_max": arm.get("graded_max"),
            "measured": measured,
            "repeat_delta": delta,
            "noise_floor": noise,
            "provenance": {"arm_id": arm["label"], "extra": arm["extra"],
                           "repeat_arm_id": rep["label"] if rep else None},
        })
    return {"units": "graded needle prefix length, 0..max", "cells": cells,
            "noise_floor": noise,
            "repeats": repeats,
            "note": "measured=false means the cell has no agreeing repeat pair; "
                    "the loss value is a single reading, not an established error"}



def build_scores_table(arms, note=None):
    """Emit the `--kv-tier-scores` grammar that src/product/kv_bit_budget.h reads.

    Why this lives here: the header's own score table is a PRIOR and says so
    ("the quality column carries the shipped priors until the automated calibration
    (tools/archkit/kv_tier_matrix.py -> JSON) supplies measured values"). This is that
    producer, so the table a scorer consumes can be the one this matrix measured
    instead of a hand-copied number.

    Two measured columns, x100, lower is better:
      speed_x100  = round((fastest_decode_tok_s / v - 1) * 100), the fastest uniform
                    arm at 0 and every slower tier scaled by its relative time cost --
                    the same normalisation the header documents for its own table.
      quality_x100= round((1 - graded_hit / graded_max) * 100), i.e. the RETRIEVAL LOSS
                    the arm actually measured. This is a needle-retrieval loss, not a
                    perplexity, and the emitted header says so: a caller that wants the
                    ctx-4096 perplexity column must supply it (that measurement is not
                    produced by this tool).
    A tier with no usable uniform arm is emitted with `# UNMEASURED` next to it and loss
    -1, never a fabricated 0: "nobody looked" and "the error is zero" must not be the
    same value (the same convention as kv_perlayer_profile.h's measured flag and this
    file's own build_w_matrix).
    """
    lines = []
    lines.append("# kv-tier-scores table emitted by tools/archkit/kv_tier_matrix.py")
    lines.append("# columns: <tier> <quality_x100> <speed_x100>, lower is better")
    lines.append(f"# {note}" if note else "# note: none given (--scores-note)")
    lines.append("# quality_x100 = retrieval loss (1 - graded_hit/graded_max) * 100 -- a NEEDLE")
    lines.append("#   loss, NOT a perplexity. The ctx-4096 ppl column is not produced here.")
    lines.append("# speed_x100   = (fastest_decode_tok_s / v - 1) * 100 over the uniform arms.")

    by_label = {a["label"]: a for a in arms}
    rates = {}
    for tier in TIERS:
        arm = by_label.get(f"uniform_{tier}")
        if arm is None:
            # --kv-dtype bf16 is spelled without a storage list; the arm label is the same.
            arm = by_label.get(f"uniform_{tier}")
        v = arm.get("decode_tok_s") if arm else None
        if v is not None:
            try:
                rates[tier] = float(v)
            except (TypeError, ValueError):
                pass
    fastest = max(rates.values()) if rates else None
    rows = []
    for tier in TIERS:
        arm = by_label.get(f"uniform_{tier}")
        v = rates.get(tier)
        if v is None or fastest is None:
            rows.append((tier, None, None))
            continue
        speed = round((fastest / v - 1.0) * 100.0)
        hit = arm.get("graded_hit")
        gmax = arm.get("graded_max") or 0
        quality = None
        if hit is not None and gmax:
            quality = round((1.0 - float(hit) / float(gmax)) * 100.0)
        rows.append((tier, quality, speed))

    out = list(lines)
    for tier, quality, speed in rows:
        if quality is None or speed is None:
            out.append(f"# {tier}: UNMEASURED (no usable uniform arm in this run) -- "
                       f"supply it or leave the header's prior in place")
            # [dl/backlog item6-producer] THE SENTINEL IS -1, WHICH IS WHAT THE DOCSTRING ABOVE
            # ALREADY PROMISES. It wrote 0 until now, and the consumer STRIPS this comment
            # (kv_bit_budget.h's parse_scores does line.find('#') then resize), so a REFUSAL
            # arrived at the DP as a MEASURED ZERO -- "nobody looked" and "the error is zero"
            # were the same value, which is the one thing this branch exists to prevent.
            # -1 is refused by the consumer's own range check, so the refusal now arrives as a
            # refusal. Measured population: 2 of 6 emitted rows in --quick on pin 8c566fba.
            out.append(f"{tier} {-1 if quality is None else quality} "
                       f"{-1 if speed is None else speed}   # UNMEASURED sentinel -1, NOT a measurement")
        else:
            out.append(f"{tier} {quality} {speed}")
    return "\n".join(out) + "\n"


def main():
    if not MODEL:
        sys.exit("KVX_MODEL must point at a .ninfer artifact (this script bakes in no default)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="只跑统一档位臂")
    # The repeat arms are what make the noise floor a measurement instead of an
    # assumption. 2 = the baseline arm and one ablation arm each run twice.
    ap.add_argument("--repeats", type=int, default=2,
                    help="how many times the noise-floor arms run (>=2 enables a floor)")
    ap.add_argument("--w-matrix-out", default=None,
                    help="also write W[layer][format] for product/kv_adapt_solver.h")
    ap.add_argument("--scores-out", default=None,
                    help="also write the --kv-tier-scores table that "
                         "src/product/kv_bit_budget.h reads, from THIS run's uniform arms")
    ap.add_argument("--scores-note", default=None,
                    help="provenance line to embed in --scores-out (what was measured)")
    ap.add_argument("--corpus", default=None,
                    help="corpus text file, for the unique_lines/degenerate guard")
    ap.add_argument("--corpus-mode", default="text", choices=["text", "corpus"])
    args = ap.parse_args()

    arms = []
    # A) 统一档位
    arms.append(("uniform_bf16", ["--kv-dtype", "bf16"]))
    for t in TIERS[1:]:
        arms.append((f"uniform_{t}", ["--kv-layer-storage", f"0-{LAYERS-1}:{t}"]))
    # D) 默认
    arms.append(("default_mixed", []))
    if not args.quick:
        # B) 单层差分（基线 nvfp4，逐层换 rk4v4）
        for l in range(LAYERS):
            arms.append((f"only_layer{l}_rk4v4",
                         ["--kv-layer-storage", spec_with("nvfp4", {l: "rk4v4"})]))
        # C) 可加性：两层同时换
        for pair in ((0, 1), (7, 8), (14, 15)):
            arms.append((f"pair_{pair[0]}_{pair[1]}_rk4v4",
                         ["--kv-layer-storage", spec_with("nvfp4", {pair[0]: "rk4v4", pair[1]: "rk4v4"})]))
        # C2) 可加性检验的 D 臂: the FACTORY 6-layer rk4v4 set changed all at once.
        #     Set S is read from the default table, not re-declared, so this arm
        #     follows the shipped table instead of a hard-coded copy of it.
        arms.append(("stack_factory_rk4v4",
                     ["--kv-layer-storage", spec_with("nvfp4", {0: "rk4v4", 1: "rk4v4", 3: "rk4v4",
                                                                4: "rk4v4", 6: "rk4v4", 7: "rk4v4"})]))

    # Noise-floor arms: the baseline and one ablation, each repeated.
    repeat_arms = [("default_mixed", [])]
    if not args.quick:
        repeat_arms.append(("only_layer0_rk4v4",
                            ["--kv-layer-storage", spec_with("nvfp4", {0: "rk4v4"})]))

    results = []
    for label, extra in arms:
        r = run_arm(label, extra)
        results.append(r)
        print(f"[{label}] rc={r['rc']} tok/s={r['decode_tok_s']} "
              f"hit={r['graded_hit']}/{r['graded_max']} payload={r['kv_payload']} "
              f"hot={r['hot']}", flush=True)

    noise = None
    for label, extra in repeat_arms:
        first = next((r for r in results if r["label"] == label), None)
        if first is None:
            continue
        for rep in range(2, max(2, args.repeats + 1)):
            r = run_arm(f"{label}_rep{rep}", extra)
            results.append(r)
            print(f"[{label}_rep{rep}] rc={r['rc']} hit={r['graded_hit']}/{r['graded_max']}",
                  flush=True)
        floor = noise_floor_of(first, results[-1])
        if floor is not None and (noise is None or floor > noise):
            noise = floor

    payload = {
        "model": MODEL,
        "needle": NEEDLE,
        "provenance": provenance_block(MODEL, MSGS, NEEDLE),
        "corpus": corpus_stats(args.corpus, args.corpus_mode) if args.corpus else None,
        "noise_floor": noise,
        "repeats": args.repeats,
        "arms": results,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")

    if args.w_matrix_out:
        w = build_w_matrix(results, args.repeats, noise)
        w["provenance"] = payload["provenance"]
        w["corpus"] = payload["corpus"]
        with open(args.w_matrix_out, "w") as fh:
            json.dump(w, fh, indent=2)
        print(f"wrote {args.w_matrix_out} ({len(w['cells'])} cells, "
              f"{sum(1 for c in w['cells'] if c['measured'])} measured, "
              f"noise_floor={noise})")

    if args.scores_out:
        note = args.scores_note or (f"binary_sha256={payload['provenance']['binary_sha256']} "
                                    f"artifact_sha256={payload['provenance']['artifact_sha256']}")
        table = build_scores_table(results, note)
        with open(args.scores_out, "w") as fh:
            fh.write(table)
        print(f"wrote {args.scores_out}")
        for line in table.splitlines():
            print("   " + line)


if __name__ == "__main__":
    main()
