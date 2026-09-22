#!/usr/bin/env python3
"""kv_budget_mirror.py -- the OFFLINE side of the KV bit-budget cross-check, AND the
adjudicator that decides what a disagreement with the engine means.

TWO MODES
  1. predict (default): solve the allocator offline and print one MIRROR| row per request.
  2. adjudicate (--engine FILE): compare the engine's own rows against the mirror's and
     print a verdict per request. A disagreement is never printed side by side and left to
     the reader -- every one of them is classified, and a real defect fails the run (exit 1).

WHAT THE OFFLINE SIDE IS
  An independent second implementation of the allocator described in
  src/product/kv_bit_budget.h: minimise the summed tier penalty subject to
  sum(bits_x100) <= round(budget*100)*layers, over L full-attention layers, with the rk4v4
  family capped at an exposure limit and an optional cold pseudo-tier (charged its real
  slot geometry, not 0).

WHY IT IS NOT tools/archkit/kv_bit_budget.py
  That file's own header banner declares its ladder to have FORKED from the authority and
  says its stdout must not be used for decisions:
      fp8  850 (authority) vs 803 (that file)
      rk4v4   425              vs 406
      iso4e 450 (pinned)     vs 300
      penalty iso4e 200 (pinned) vs 50
  Regenerating the matrix from it is exactly how the shipped "request -> achieved -> spec"
  table came to describe a ladder the engine does not have: every shipped row reproduces
  that tool. This mirror reads the ladder from the SAME dump the probe emits, so a fork
  cannot be reintroduced silently (check C1).

HONEST LIMITS (why a spec difference is not automatically a bug)
  * cold_cap > 0: the engine deliberately keeps this same multiset DP
    (kv_bit_budget_solve -> kv_bit_budget_solve_impl). Here the mirror is a true twin and
    ANY difference is a defect (C5a).
  * cold_cap == 0 (the DEFAULT, `--max-cold-pages 0`): the engine hands the solve to the
    layer-exact "gearbox" solver. A multiset DP cannot name a non-prefix rk4v4 set, so a
    SPEC difference there is expected. What is still a defect is the engine leaving
    something on the table: a strictly higher penalty at the same ceiling (C5c), or a
    refusal where the mirror fits (C5d).
"""
from __future__ import annotations

import argparse
import json
import sys

RK4V4_ROW_NAMES = ("rk4v4", "rk3v4", "rk2v4")
PACK_ORDER = ["rk4v4", "bf16", "int8", "fp8", "nvfp4", "iso4e"]
PACK_ORDER_COLD = ["rk4v4", "cold", "bf16", "int8", "fp8", "nvfp4", "iso4e"]
TOL = 1e-6
GRID = 1e-6

TIERS: dict[str, int] = {}
PENALTY: dict[str, int] = {}
LADDER_ROWS: list[tuple[str, int, int, bool]] = []
# NOT DEFAULTS. These three are SENTINELS meaning "nothing has been read from the probe's dump
# yet"; main() overwrites every one of them from the dump or REFUSES to run at all. Until
# 2026-09-18 they held plausible-looking values (327 / 25 / 8) and main() read them with
# `consts.get(key, COLD_BITS_X100)`, so a dump carrying no CONST|cold_bits_x100 row was
# ADJUDICATED AT 327 -- the retired 2.60-bit rANS grid point -- inside a tool whose stated rule
# (HONEST LIMITS, C5a) is that with cold_cap > 0 a difference of ANY size is a defect. What made
# that silent is that the two cases are indistinguishable in the output: a dump that says 327 and
# a dump that says nothing both printed `cold_bits_x100=327`. The sentinel is deliberately not a
# plausible grid point, so a missed assignment shows up as nonsense instead of as agreement.
AUTHORITY_NOT_READ = -1
COLD_BITS_X100 = AUTHORITY_NOT_READ
COLD_PENALTY_X100 = AUTHORITY_NOT_READ
RK4V4_LAYER_LIMIT = AUTHORITY_NOT_READ
# The dump keys those three are read from, spelled once so the read and the refusal cannot drift.
# `kKvBitBudgetE8LayerLimit` is the one a command line may supply instead (--rk4v4-limit).
AUTHORITY_KEYS = ("cold_bits_x100", "cold_penalty_x100", "kKvBitBudgetE8LayerLimit")

DEFECTS: list[str] = []


def defect(msg: str) -> None:
    DEFECTS.append(msg)
    print(f"DEFECT|{msg}")


def load_ladder(path: str) -> list[tuple[str, int, int, bool]]:
    rows: list[tuple[str, int, int, bool]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("LADDER|"):
                continue
            parts = line.split("|")
            if parts[1] == "tier":
                continue
            rows.append((parts[1], int(parts[2]), int(parts[3]), bool(int(parts[4]))))
    return rows


def load_consts(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("CONST|"):
                _, key, value = line.split("|", 2)
                out[key] = value
    return out


def solve(layers: int, budget_bits: float, rk4v4_limit: int, cold_cap: int,
          criterion: str = "lexicographic", gears: tuple[str, ...] | None = None):
    """Multiset DP. Returns (penalty_x100, bits_x100, counts).

    criterion:
      'lexicographic' -- THE ENGINE'S CURRENT OBJECTIVE (kv_bit_budget.h, "SATURATION
          (BUDGETSAT): an incumbent is judged on the SAME lexicographic order the DP now
          optimises -- MORE BITS first, penalty only on an equal-bit tie"): maximise the
          achieved bits inside the ceiling, then minimise the penalty. This is the default
          because it is what the header declares, and aregen that compares against the
          other one would report the engine's deliberate change as a bug.
      'penalty-min'   -- the objective the SHIPPED table and this doc's section 3 state
          ("min sum(penalty_x100) s.t. sum(bits_x100) <= budget*100*layers"). Kept as a
          second reference so the trade the new criterion makes stays visible.
    """
    capacity = int(round(budget_bits * 100)) * layers
    order = [name for name, _, _, sel in LADDER_ROWS if sel]
    if gears:
        # F_l -- the engine's PER-LAYER ADMISSIBLE GEAR SET. Without this the mirror keeps
        # beating the engine with a gear the engine's own constraint excludes (fp8 under the
        # default component mode: the gearbox drops ladder index 2 unless rotation is on),
        # and every such row would be reported as a defect the engine is not guilty of.
        order = [name for name in order if name in set(gears)]
    dp = {(0, 0): (0, {})}
    for _ in range(layers):
        nxt: dict[tuple[int, int], tuple[int, dict[str, int]]] = {}
        for (bits, cold_used), (penalty, counts) in dp.items():
            rk4v4_used = sum(counts.get(n, 0) for n in RK4V4_ROW_NAMES)
            for tier in order:
                if tier in RK4V4_ROW_NAMES and rk4v4_used >= rk4v4_limit:
                    continue
                new_bits = bits + TIERS[tier]
                if new_bits > capacity:
                    continue
                new_counts = dict(counts)
                new_counts[tier] = new_counts.get(tier, 0) + 1
                key = (new_bits, cold_used)
                cand = (penalty + PENALTY[tier], new_counts)
                if key not in nxt or cand[0] < nxt[key][0]:
                    nxt[key] = cand
            if cold_cap > 0 and cold_used < cold_cap:
                new_bits = bits + COLD_BITS_X100
                if new_bits <= capacity:
                    new_counts = dict(counts)
                    new_counts["cold"] = new_counts.get("cold", 0) + 1
                    key = (new_bits, cold_used + 1)
                    cand = (penalty + COLD_PENALTY_X100, new_counts)
                    if key not in nxt or cand[0] < nxt[key][0]:
                        nxt[key] = cand
        dp = nxt
        if not dp:
            raise ValueError("no feasible allocation")
    # dp is keyed on (bits, cold_used); every key was reached in exactly `layers` steps.
    if criterion == "lexicographic":
        # MORE BITS first, penalty only on an equal-bit tie.
        best_bits = max(bits for bits, _cold in dp)
        best = None
        for (bits, _cold), (penalty, counts) in dp.items():
            if bits != best_bits:
                continue
            if best is None or penalty < best[0]:
                best = (penalty, bits, counts)
    else:
        best = None
        for (bits, _cold), (penalty, counts) in dp.items():
            if best is None or penalty < best[0]:
                best = (penalty, bits, counts)
    if best is None:
        raise ValueError("no feasible allocation")
    return best


def to_spec(counts: dict[str, int], layers: int) -> str:
    pack = PACK_ORDER_COLD if counts.get("cold", 0) else PACK_ORDER
    slots: list[str | None] = [None] * layers
    cursor = 0
    for tier in pack:
        for _ in range(counts.get(tier, 0)):
            slots[cursor] = tier
            cursor += 1
    parts: list[str] = []
    begin = 0
    while begin < layers:
        tier = slots[begin]
        end = begin
        while end + 1 < layers and slots[end + 1] == tier:
            end += 1
        parts.append(f"{begin}:{tier}" if begin == end else f"{begin}-{end}:{tier}")
        begin = end + 1
    return ",".join(parts)


def expand_spec(spec: str, lanes: int) -> list[str]:
    out: list[str | None] = [None] * lanes
    for item in [p for p in spec.split(",") if p]:
        left, tier = item.split(":")
        if "-" in left:
            lo, hi = left.split("-")
            lo, hi = int(lo), int(hi)
        else:
            lo = hi = int(left)
        for i in range(lo, hi + 1):
            out[i] = tier
    return [t for t in out if t is not None]  # type: ignore[return-value]


def spec_cost(spec: str, lanes: int) -> tuple[float, float]:
    """Recompute (achieved_bits, penalty) from a spec line alone, using the ENGINE's ladder."""
    tiers = expand_spec(spec, lanes)
    if len(tiers) != lanes:
        raise ValueError(f"spec covers {len(tiers)} of {lanes} layers: {spec}")
    bits = 0
    pen = 0
    for t in tiers:
        if t == "cold":
            bits += COLD_BITS_X100
            pen += COLD_PENALTY_X100
        else:
            if t not in TIERS:
                raise ValueError(f"spec names a tier not on the ladder: {t}")
            bits += TIERS[t]
            pen += PENALTY[t]
    return bits / (lanes * 100.0), pen / 100.0


def parse_engine(path: str) -> dict[tuple[float, int, str], dict]:
    rows: dict[tuple[float, int, str], dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ROW|"):
                continue
            fields = dict(p.split("=", 1) for p in line.strip().split("|")[1:] if "=" in p)
            key = (round(float(fields["request"]), 4), int(fields["cold_cap"]),
                   fields["spelling"])
            rows[key] = fields
    return rows


def parse_mirror(path: str) -> dict[tuple[float, int], dict]:
    rows: dict[tuple[float, int], dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("MIRROR|"):
                continue
            fields = dict(p.split("=", 1) for p in line.strip().split("|")[1:] if "=" in p)
            rows[(round(float(fields["request"]), 4), int(fields["cold_cap"]))] = fields
    return rows


def adjudicate(engine_path: str, mirror_path: str, layers: int, cold_cap: int,
               criterion: str = "lexicographic") -> int:
    eng = parse_engine(engine_path)
    mir = parse_mirror(mirror_path)
    requests = sorted({k[0] for k in eng} | {k[0] for k in mir})
    print(f"# adjudication: {len(requests)} request(s), {layers} layers, cold_cap={cold_cap}")
    print(f"# the engine's DECLARED objective here is '{criterion}' (see the mirror's --criterion)")
    print("# the comparison is on (achieved_bits, penalty), the two quantities the report prints.")
    print("# The spec's layer MEMBERSHIP is NOT determined by the objective -- two plans with the")
    print("# same tier multiset differ only in WHICH layer gets which tier, and the header's own")
    print("# 'a SET, not a COUNT' argument says that placement is a separate, open question. A")
    print("# spec difference at equal (achieved, penalty) is therefore reported, not failed.")
    print("# verdicts: AGREE / EQUIVALENT / EXPECTED-DIFF / OBJECTIVE-DIFF / DEFECT")
    for req in requests:
        j = eng.get((req, cold_cap, "kv-bits"))
        b = eng.get((req, cold_cap, "kv-bit-budget"))
        m = mir.get((req, cold_cap))
        notes: list[str] = []
        verdict = "AGREE"

        eng_deployed = j is not None and j.get("spec") is not None
        eng_refused = j is not None and j.get("spec") is None

        # ---- C2: the two flag spellings must resolve to the same allocator result ----
        if j is not None and b is not None:
            b_deployed = b.get("spec") is not None
            if eng_deployed and b_deployed:
                if (j["spec"] != b["spec"] or abs(float(j["achieved"]) - float(b["achieved"])) > TOL
                        or abs(float(j["penalty"]) - float(b["penalty"])) > TOL):
                    defect(f"C2 request={req} cold_cap={cold_cap}: --kv-bits and "
                           f"--kv-bit-budget disagree: "
                           f"{j['spec']} (ach {j['achieved']}) vs {b['spec']} (ach {b['achieved']})")
                    verdict = "DEFECT"
            elif eng_deployed != b_deployed:
                # A refusal raised at a different layer is a message difference, not a
                # numeric one; only a split verdict is a defect.
                defect(f"C2 request={req} cold_cap={cold_cap}: one spelling deployed and the "
                       f"other refused (kv-bits deployed={eng_deployed}, "
                       f"kv-bit-budget deployed={b_deployed})")
                verdict = "DEFECT"
            else:
                notes.append("both spellings refuse (refusal text differs by call site, "
                             "which is a message difference, not a numeric one)")

        # ---- C3: achieved must be the plain per-layer average of the spec line ----
        if eng_deployed:
            spec_achieved, spec_penalty = spec_cost(j["spec"], layers)
            if abs(spec_achieved - float(j["achieved"])) > 1e-4:
                defect(f"C3 request={req} cold_cap={cold_cap}: reported achieved="
                       f"{j['achieved']} != average recomputed from its own spec="
                       f"{spec_achieved:.4f} (spec {j['spec']})")
                verdict = "DEFECT"
            if abs(spec_penalty - float(j["penalty"])) > 1e-4:
                defect(f"C3 request={req} cold_cap={cold_cap}: reported penalty="
                       f"{j['penalty']} != sum recomputed from its own spec="
                       f"{spec_penalty:.4f}")
                verdict = "DEFECT"

        # ---- C4: the request must be respected (no overrun) ----
        if eng_deployed and float(j["achieved"]) > req + GRID:
            defect(f"C4 request={req} cold_cap={cold_cap}: achieved={j['achieved']} EXCEEDS the "
                   f"ceiling")
            verdict = "DEFECT"

        # ---- C5: engine vs the offline mirror, on the DECLARED criterion ----
        if m is None:
            notes.append("mirror produced no row")
        elif m.get("spec") is None:
            if eng_deployed:
                notes.append(f"mirror refuses where the engine deploys {j['spec']} "
                             f"(mirror is the weaker solver here; not a defect)")
                if verdict == "AGREE":
                    verdict = "EXPECTED-DIFF"
        elif not eng_deployed:
            if float(m["achieved"]) <= req + GRID and eng_refused:
                defect(f"C5d request={req} cold_cap={cold_cap}: engine REFUSED but the offline "
                       f"mirror fits {m['spec']} at achieved={m['achieved']}")
                verdict = "DEFECT"
        else:
            same_spec = j["spec"] == m["spec"]
            d_pen = float(j["penalty"]) - float(m["penalty"])
            d_ach = float(j["achieved"]) - float(m["achieved"])
            # Under 'lexicographic' the mirror maximises bits too, so more engine bits is a
            # win for the engine, not a defect; under 'penalty-min' the mirror only fits the
            # ceiling, so a lower engine attained-bits is a win for the engine.
            if abs(d_ach) > 1e-4:
                if d_ach > 0 and criterion == "lexicographic":
                    notes.append(f"engine attains MORE bits than the mirror "
                                 f"({j['achieved']} vs {m['achieved']}) at this ceiling")
                elif d_ach < 0 and criterion == "penalty-min":
                    notes.append(f"engine attains FEWER bits than the mirror "
                                 f"({j['achieved']} vs {m['achieved']}) but satisfies the "
                                 f"ceiling; not a defect under this criterion")
                else:
                    defect(f"C5 request={req} cold_cap={cold_cap}: engine achieved_bits "
                           f"{j['achieved']} != mirror {m['achieved']} under the '{criterion}' "
                           f"criterion (engine {j['spec']} vs mirror {m['spec']})")
                    verdict = "DEFECT"
            if d_pen > TOL:
                if criterion == "lexicographic" and abs(d_ach) <= 1e-4:
                    defect(f"C5 request={req} cold_cap={cold_cap}: at EQUAL bits the engine "
                           f"penalty {j['penalty']} is above the mirror's {m['penalty']} "
                           f"(engine {j['spec']} vs mirror {m['spec']})")
                    verdict = "DEFECT"
                elif criterion != "lexicographic":
                    defect(f"C5 request={req} cold_cap={cold_cap}: engine penalty {j['penalty']} "
                           f"is ABOVE the offline mirror's {m['penalty']} at the same ceiling "
                           f"(engine {j['spec']} vs mirror {m['spec']})")
                    verdict = "DEFECT"
                else:
                    notes.append(f"engine trades penalty for bits: engine pen={j['penalty']} "
                                 f"at ach={j['achieved']} vs mirror pen={m['penalty']} at "
                                 f"ach={m['achieved']} -- the declared lexicographic order "
                                 f"prefers the bits, so this is the documented trade, not a bug")
                    if verdict == "AGREE":
                        verdict = "OBJECTIVE-DIFF"
            if same_spec and verdict == "AGREE":
                notes.append("identical plan")
            elif not same_spec and verdict == "AGREE":
                notes.append(f"same (achieved, penalty), different layer membership: engine "
                             f"{j['spec']} vs mirror {m['spec']}")
                verdict = "EQUIVALENT"
        print(f"VERDICT|request={req:.2f}|cold_cap={cold_cap}|{verdict}"
              + (f"|notes={' ; '.join(notes)}" if notes else ""))

    print(f"# defects: {len(DEFECTS)}")
    return 1 if DEFECTS else 0


def main() -> int:
    global TIERS, PENALTY, RK4V4_LAYER_LIMIT, COLD_BITS_X100, COLD_PENALTY_X100, LADDER_ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-file", required=True,
                    help="the probe's `ladder` output; the ladder is READ FROM HERE, never assumed")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--bits", type=float, nargs="*", default=[])
    ap.add_argument("--cold-cap", type=int, default=0)
    ap.add_argument("--rk4v4-limit", type=int, default=None)
    ap.add_argument("--gears", default="",
                    help="comma list of the engine's PER-LAYER ADMISSIBLE gear set (F_l); "
                         "empty = every selectable ladder row. The probe's refusal text names "
                         "it ('candidate space at layer 0: ...'), which is where the driver "
                         "reads it from rather than re-deriving it.")
    ap.add_argument("--criterion", default="lexicographic",
                    choices=("lexicographic", "penalty-min"),
                    help="'lexicographic' = the engine's declared objective (more bits first, "
                         "penalty on ties); 'penalty-min' = the objective the shipped table "
                         "states")
    ap.add_argument("--engine", default="",
                    help="the probe's rows file: switches the tool to adjudication mode")
    ap.add_argument("--mirror-rows", default="",
                    help="the mirror's own rows file to adjudicate against (with --engine)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    LADDER_ROWS = load_ladder(args.ladder_file)
    if not LADDER_ROWS:
        print("mirror: no LADDER| rows found -- refusing to guess a ladder", file=sys.stderr)
        return 2
    consts = load_consts(args.ladder_file)
    TIERS = {name: bits for name, bits, _, _ in LADDER_ROWS}
    PENALTY = {name: pen for name, _, pen, _ in LADDER_ROWS}

    # C1, FIRST HALF: the numbers this tool adjudicates AGAINST must be the engine's own, or the
    # tool does not run. A key the dump does not carry is REFUSED, never defaulted -- see
    # AUTHORITY_NOT_READ above for what stood here and what it cost. This does NOT change what a
    # DIFFERENCE means: a C5 disagreement is still classified, still printed, still fatal. What is
    # different in kind is a tool declining to adjudicate against a value it never read.
    required = list(AUTHORITY_KEYS)
    if args.rk4v4_limit is not None:
        required.remove("kKvBitBudgetE8LayerLimit")
    missing = [k for k in required if k not in consts]
    if missing:
        # The escape hatch is offered ONLY when it is the whole story. `--rk4v4-limit` supplies
        # kKvBitBudgetE8LayerLimit and nothing else, so a message that offered it beside a
        # missing cold_bits_x100 would be pointing the reader at a flag that cannot help -- the
        # same class of false promise this refusal exists to end.
        hint = (", or pass --rk4v4-limit explicitly."
                if missing == ["kKvBitBudgetE8LayerLimit"]
                else ". There is no command-line equivalent for the cold grid point or the cold "
                     "penalty: they have to come from the dump.")
        print("mirror: the dump " + args.ladder_file + " carries no "
              + ", ".join("CONST|" + k for k in missing)
              + " row. These ARE the authority this tool adjudicates against -- "
                "kKvBitBudgetColdBitsX100 (derived from kKvColdPoolStrideBytes, "
                "src/product/kv_bit_budget.h), kKvBitBudgetColdPenaltyX100 and "
                "kKvBitBudgetE8LayerLimit -- and standing a local default in for one of them is "
                "how a dump with no cold_bits_x100 row came to be adjudicated at the retired "
                "2.60-bit grid point 327. Re-run tools/archkit/kv_budget_probe.cpp so the dump "
                "carries the CONST rows" + hint,
              file=sys.stderr)
        return 2
    COLD_BITS_X100 = int(consts["cold_bits_x100"])
    COLD_PENALTY_X100 = int(consts["cold_penalty_x100"])
    rk4v4_limit = (args.rk4v4_limit if args.rk4v4_limit is not None
                   else int(consts["kKvBitBudgetE8LayerLimit"]))

    # C1: the ladder the mirror fits against must be the engine's ladder, and the mirror must
    # be able to prove it read it rather than carrying a private copy.
    selectable = [n for n, _, _, s in LADDER_ROWS if s]
    gears = tuple(g for g in (args.gears.split(",") if args.gears else []) if g)
    print(f"# C1 ladder from {args.ladder_file}: {len(LADDER_ROWS)} rows, "
          f"{len(selectable)} selectable ({', '.join(selectable)}) "
          f"| cold_bits_x100={COLD_BITS_X100} cold_penalty_x100={COLD_PENALTY_X100} "
          f"rk4v4_limit={rk4v4_limit}")
    # WHERE each of the three CAME FROM, stated so that "read from the dump" can never again be
    # mistaken for "filled in from this file". A value that was not read does not reach here: the
    # refusal above returns before this line.
    print(f"# C1 authority source: cold_bits_x100 and cold_penalty_x100 READ from the dump's "
          f"CONST| rows; rk4v4_limit "
          + ("from --rk4v4-limit"
             if args.rk4v4_limit is not None
             else "READ from the dump's CONST|kKvBitBudgetE8LayerLimit") +
          " -- none of the three is a local default")
    print(f"# F_l admissible gear set: {', '.join(gears) if gears else '<every selectable row>'}"
          + (f"  (EXCLUDED: {', '.join(g for g in selectable if g not in gears)})"
             if gears else ""))
    if "rk4v4" not in TIERS or "iso4e" not in TIERS:
        defect("C1 the ladder is missing rk4v4 or iso4e -- the dump is not a usable ladder")

    emitted = []
    if not args.engine:
        for request in args.bits:
            try:
                penalty_x100, bits_x100, counts = solve(args.layers, request, rk4v4_limit,
                                                        args.cold_cap, args.criterion, gears)
            except ValueError:
                print(f"MIRROR|request={request:.2f}|cold_cap={args.cold_cap}|REFUSED")
                continue
            spec = to_spec(counts, args.layers)
            mix = "+".join(f"{k}x{v}" for k, v in sorted(counts.items()))
            print(f"MIRROR|request={request:.2f}|cold_cap={args.cold_cap}"
                  f"|achieved={bits_x100 / (args.layers * 100):.4f}"
                  f"|penalty={penalty_x100 / 100.0:.4f}|spec={spec}|mix={mix}"
                  f"|criterion={args.criterion}")
            emitted.append({"request": request, "cold_cap": args.cold_cap,
                            "achieved": bits_x100 / (args.layers * 100),
                            "penalty": penalty_x100 / 100.0, "spec": spec, "mix": mix,
                            "criterion": args.criterion})
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(emitted, handle, indent=2)
        # Predict mode used to `return 0` unconditionally, which made the C1 ladder defect it
        # prints above invisible to the shell: tools/archkit/kv_budget_regen.sh:144-148 wraps
        # this very invocation in `if ! python3 "$MIRROR_SRC" ... ; then echo "MIRROR FAILED";
        # exit 4; fi`, so the caller's contract already IS "non-zero means the mirror could not
        # do its job". adjudicate() has always returned `1 if DEFECTS else 0`; predict mode now
        # does the same, so a `DEFECT|` line and a non-zero exit are never separated again.
        print(f"# defects: {len(DEFECTS)}")
        return 1 if DEFECTS else 0

    rc = adjudicate(args.engine, args.mirror_rows, args.layers, args.cold_cap, args.criterion)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
