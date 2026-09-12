#!/usr/bin/env python3
"""Decisive probe: is the FlashNext contract one prefix-canonicalisation away
from full coverage of the real checkpoint?

Measured: `audit-gguf` (and a set-based reimplementation of the same semantics)
matches 1 / 296,475 real keys, because the checkpoint names everything under
`model.language_model.*` while the contract only knows `model.*` (HF) and
`blk.*` (llama.cpp). This applies candidate canon normalisations and reports
coverage for each, plus the NVFP4-quant companion story.

Speed: the naive audit is keys x entries x aliases (hours). Here every alias is
resolved either by set lookup (concrete) or, if it contains a glob, by scanning
only the sorted-key range that shares its literal prefix (bisect).
"""
from __future__ import annotations

import bisect
import fnmatch
import sys
from pathlib import Path

ARCH = Path(r"C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo\tools\archkit")
NAMES = Path(r"C:\Users\User\Documents\ziqinzhang\_collab\M_flashnext_names.txt")
OUT = Path(r"C:\Users\User\Documents\ziqinzhang\_collab\M_flashnext_prefix_probe.md")

sys.path.insert(0, str(ARCH))
import flashnext_bindings as fb                                   # noqa: E402

lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def coverage(names: list[str], entries) -> tuple[int, int]:
    ordered = sorted(names)
    name_set = set(names)
    hit_entries = 0
    consumed: set[str] = set()
    for e in entries:
        hit = False
        for pat in e.get("alias") or []:
            glob_at = min((pat.find(c) for c in "*?[" if c in pat), default=-1)
            if glob_at < 0:
                if pat in name_set:
                    hit = True
                    consumed.add(pat)
            else:
                lit = pat[:glob_at]
                lo = bisect.bisect_left(ordered, lit)
                for key in ordered[lo:]:
                    if not key.startswith(lit):
                        break
                    if key not in consumed and fnmatch.fnmatch(key, pat):
                        hit = True
                        consumed.add(key)
        hit_entries += hit
    return hit_entries, len(consumed)


def canon(n: str, mode: str) -> str:
    if mode == "orig":
        return n
    if mode == "strip_lm":          # model.language_model.X -> model.X
        return "model." + n[len("model.language_model."):] if n.startswith("model.language_model.") else n
    if mode == "drop_model":        # model.language_model.X -> language_model.X
        return n[len("model."):] if n.startswith("model.") else n
    raise ValueError(mode)


def main() -> int:
    raw = [n for n in NAMES.read_text(encoding="utf-8").splitlines() if n]
    say("# FlashNext 契约 vs 真实 checkpoint: 前缀规范化探针")
    say()
    say("- 真实键 %d, 契约条目 %d" % (len(raw), len(fb.E)))
    pref: dict[str, int] = {}
    for n in raw:
        key = ".".join(n.split(".")[:2])
        pref[key] = pref.get(key, 0) + 1
    say("- 前两段分布 Top6: %s" % sorted(pref.items(), key=lambda kv: -kv[1])[:6])
    say()
    say("| 规范化 | 命中条目 | 消费键 |")
    say("|---|---|---|")
    for mode in ("orig", "strip_lm", "drop_model"):
        names = [canon(n, mode) for n in raw]
        he, cons = coverage(names, fb.E)
        say("| %s | %d / %d | %d / %d |" % (mode, he, len(fb.E), cons, len(raw)))

    say()
    raw_set = set(raw)
    quant = (".weight_scale", ".weight_scale_2", ".input_scale")
    weights = [n for n in raw if n.endswith(".weight")]
    complete = sum(1 for w in weights
                   if all((w[:-len(".weight")] + q) in raw_set for q in quant))
    say("- 真实 `.weight` %d 个; NVFP4 四元组 (weight/scale/scale_2/input_scale) 齐全的 %d 个"
        % (len(weights), complete))
    say("- 契约条目里含 scale/quant 字样的: %d" %
        sum(1 for e in fb.E if any(q.strip(".") in str(e) for q in quant)))
    say()
    say("## 判读")
    say("- 若 `strip_lm` 接近满覆盖 ⇒ 契约只差一层前缀规范化 (`model.language_model.`→`model.`);")
    say("  修法首选转换器入口规范化 (不动 74,520 条 alias), 其次才是逐条加前缀变体。")
    say("- NVFP4 伴生张量若不在契约里, 转换器必须自行推导/消费, 否则权重会被静默丢弃。")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwritten:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
