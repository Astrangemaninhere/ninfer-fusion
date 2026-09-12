#!/usr/bin/env python3
"""Fast, semantics-preserving audit of the FlashNext bindings contract.

`flashnext_bindings.py --audit-gguf` nests three loops -- 296,475 checkpoint keys x
~74,520 contract entries x aliases -- and fnmatch is pure Python, so that is ~2e10
comparisons (hours; measured 2,339 CPU-seconds before it was killed). The matching
semantics are the same here (pattern = alias, subject = key), but the concrete
aliases are resolved through a set lookup and only the few globbed aliases fall
back to fnmatch.

Reports the two directions the contract promises:
  * every checkpoint key is consumed by some entry  (nothing silently dropped)
  * every entry found a source key                  (nothing silently missing)
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

ARCH = Path(r"C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo\tools\archkit")
NAMES = Path(r"C:\Users\User\Documents\ziqinzhang\_collab\M_flashnext_names.txt")
OUT = Path(r"C:\Users\User\Documents\ziqinzhang\_collab\M_flashnext_fast_audit.md")

sys.path.insert(0, str(ARCH))
import flashnext_bindings as fb                                   # noqa: E402

lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def main() -> int:
    names = [n for n in NAMES.read_text(encoding="utf-8").splitlines() if n]
    name_set = set(names)
    entries = fb.E
    say("# FlashNext 契约审计 (快版, 语义同 flashnext_bindings.audit)")
    say()
    say("- checkpoint 键: %d  |  契约条目: %d" % (len(names), len(entries)))

    globbed = 0
    entry_hit: set[str] = set()
    consumed: set[str] = set()
    missing: list[str] = []
    for e in entries:
        engine = e["engine"]
        aliases = e.get("alias") or []
        hit = False
        for pat in aliases:
            if any(c in pat for c in "*?["):
                globbed += 1
                for key in names:
                    if key in consumed:
                        continue
                    if fnmatch.fnmatch(key, pat):
                        hit = True
                        entry_hit.add(engine)
                        consumed.add(key)
            else:
                if pat in name_set:
                    hit = True
                    entry_hit.add(engine)
                    consumed.add(pat)
        if not hit:
            missing.append(engine)

    unconsumed = [n for n in names if n not in consumed]
    say("- 含通配的 alias 回退次数: %d (其余走 set 查表)" % globbed)
    say()
    say("## 方向 1: 契约条目 -> 源键 (缺失即引擎张量没有权重来源)")
    say("- 命中条目: %d / %d" % (len(entry_hit), len(entries)))
    say("- 未命中条目: %d" % len(missing))
    for m in missing[:25]:
        say("  - %s" % m)
    if len(missing) > 25:
        say("  ... 另有 %d 条" % (len(missing) - 25))
    say()
    say("## 方向 2: 源键 -> 契约 (未消费即被静默丢弃)")
    say("- 已消费: %d / %d" % (len(consumed), len(names)))
    say("- 未消费: %d" % len(unconsumed))
    by_suffix: dict[str, int] = {}
    for n in unconsumed:
        tail = n.rsplit(".", 1)[-1]
        by_suffix[tail] = by_suffix.get(tail, 0) + 1
    for k, v in sorted(by_suffix.items(), key=lambda kv: -kv[1])[:20]:
        say("  - .%s x%d" % (k, v))
    for n in unconsumed[:15]:
        say("  e.g. %s" % n)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwritten:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
