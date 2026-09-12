#!/usr/bin/env python3
"""Settle the dflash2 alignment question empirically.

Inputs
  dl/df2_probe_raw.log   NINFER_DF2DBG=1 stderr from the dflash2 run
  dl/df2_plain.log       --print-token-ids greedy run, NO spec (ground truth)

Questions answered, in order of decisiveness:
  Q1 target side: is the verify block's argmax stream exactly the target's own greedy
     stream? For each round, find the offset t in the plain stream with
     plain[t] == argmax[0] and count how many columns then continue to match.
     If Q1 fails everywhere, the verify-side target forward (positions / KV / mask)
     is wrong, and the draft never had a chance.
  Q2 draft side: how often does argmax[c] == draft[c] per position? Compare that
     against the engine's own reported `accepted` count. If the measured agreement
     is much higher than `accepted`, the acceptance kernel compares the wrong
     columns; if it matches, the draft block itself is what is wrong.
"""

import pathlib
import re
import sys

ROUND = re.compile(
    r"\[df2dbg\] row=(?P<row>-?\d+) anchor=(?P<anchor>-?\d+) frontier=(?P<frontier>-?\d+) "
    r"valid=(?P<valid>-?\d+) in_extent=(?P<in_extent>-?\d+) out_extent=(?P<out_extent>-?\d+) "
    r"accepted=(?P<accepted>-?\d+) count=(?P<count>-?\d+)")
COL = re.compile(
    r"\[df2dbg\]\s+col=(?P<col>-?\d+) pos=(?P<pos>-?\d+) verify=(?P<verify>-?\d+) "
    r"draft=(?P<draft>-?\d+) argmax=(?P<argmax>-?\d+)")


def parse(path):
    rounds = []
    current = None
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = ROUND.search(line)
        if m:
            current = {"row": int(m.group("row")), "anchor": int(m.group("anchor")),
                       "valid": int(m.group("valid")), "in_extent": int(m.group("in_extent")),
                       "out_extent": int(m.group("out_extent")),
                       "accepted": int(m.group("accepted")), "count": int(m.group("count")),
                       "cols": []}
            rounds.append(current)
            continue
        m = COL.search(line)
        if m and current is not None:
            current["cols"].append({k: int(m.group(k))
                                    for k in ("col", "pos", "verify", "draft", "argmax")})
    return rounds


def plain_ids(path):
    ids = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("tokens") and "generated ids" in line:
            ids = [int(x) for x in line.split("generated ids", 1)[1].split()]
    return ids


def main():
    probe = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_probe_raw.log")
    plainp = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/df2_plain.log")
    rounds = parse(probe)
    plain = plain_ids(plainp) if plainp.exists() else []
    print(f"rounds parsed: {len(rounds)}   plain greedy ids: {len(plain)}")
    if not rounds:
        print("NO_PROBE_DATA")
        return 1

    drafting = [r for r in rounds if r["in_extent"] > 0]
    print(f"rounds that drafted anything (in_extent>0): {len(drafting)}")
    print()

    # ---------------- Q2: argmax vs draft, per position ----------------
    width = max(len(r["cols"]) for r in rounds)
    agree = [0] * width
    seen = [0] * width
    accepted_by_pos = [0] * width
    for r in rounds:
        if r["in_extent"] <= 0:
            continue
        for c in r["cols"]:
            i = c["col"]
            if i >= width:
                continue
            if c["draft"] == -1:
                continue                      # last column has no draft
            seen[i] += 1
            if c["argmax"] == c["draft"]:
                agree[i] += 1
        for i in range(min(r["accepted"], width)):
            accepted_by_pos[i] += 1

    print("Q2  per position: measured argmax==draft  vs  engine-reported accepted")
    print(f"    {'pos':>4} {'measured':>9} {'engine':>7} {'n':>4} {'measured%':>10} {'engine%':>8}")
    tot_m = tot_e = tot_n = 0
    for i in range(width - 1):
        pct_m = 100.0 * agree[i] / seen[i] if seen[i] else 0.0
        pct_e = 100.0 * accepted_by_pos[i] / seen[i] if seen[i] else 0.0
        print(f"    {i:>4} {agree[i]:>9} {accepted_by_pos[i]:>7} {seen[i]:>4} "
              f"{pct_m:>9.2f}% {pct_e:>7.2f}%")
        tot_m += agree[i]; tot_e += accepted_by_pos[i]; tot_n += seen[i]
    print(f"    {'ALL':>4} {tot_m:>9} {tot_e:>7} {tot_n:>4} "
          f"{(100.0*tot_m/tot_n if tot_n else 0):>9.2f}% "
          f"{(100.0*tot_e/tot_n if tot_n else 0):>7.2f}%")
    print()

    # ---------------- Q1: is the verify argmax the target's greedy stream? ----------------
    if not plain:
        print("Q1  skipped: no ground-truth stream parsed from dl/df2_plain.log")
    else:
        pos_index = {}
        for t, tok in enumerate(plain):
            pos_index.setdefault(tok, []).append(t)
        full = partial = none = 0
        details = []
        for r in drafting:
            cols = r["cols"]
            if not cols:
                continue
            g0 = cols[0]["argmax"]
            cands = pos_index.get(g0, [])
            best = (-1, -1)               # (offset, matched columns)
            for t in cands:
                matched = 0
                for c in cols:
                    if t + c["col"] < len(plain) and plain[t + c["col"]] == c["argmax"]:
                        matched += 1
                    else:
                        break
                if matched > best[1]:
                    best = (t, matched)
            ncols = sum(1 for c in cols if c["draft"] != -1)
            if best[1] >= ncols:
                full += 1
            elif best[1] >= 1:
                partial += 1
            else:
                none += 1
            details.append((r["anchor"], r["in_extent"], best[0], best[1], ncols,
                            [c["argmax"] for c in cols]))
        print("Q1  does the verify argmax stream reproduce the no-spec greedy stream?")
        print(f"    rounds landing exactly on the stream : {full}")
        print(f"    rounds matching only a prefix        : {partial}")
        print(f"    rounds with no match at all          : {none}")
        print()
        print("    first 8 drafting rounds: anchor in_extent offset matched/ncols argmax[]")
        for row in details[:8]:
            print(f"      anchor={row[0]:<7} in_extent={row[1]} offset={row[2]:<7} "
                  f"{row[3]}/{row[4]}  {row[5]}")
        print()
        print("    first 24 ground-truth ids:", plain[:24])

    print()
    print("SAMPLE first drafting round, raw columns:")
    for r in drafting[:1]:
        print(f"  anchor={r['anchor']} valid={r['valid']} in_extent={r['in_extent']} "
              f"accepted={r['accepted']} count={r['count']}")
        for c in r["cols"]:
            flag = "AGREE" if c["argmax"] == c["draft"] else ""
            print(f"    col={c['col']} pos={c['pos']} verify={c['verify']:<8} "
                  f"draft={c['draft']:<8} argmax={c['argmax']:<8} {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
