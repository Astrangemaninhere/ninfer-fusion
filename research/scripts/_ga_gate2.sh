#!/bin/bash
# _ga_gate2.sh -- G-A gate (greedy token-id consistency).  v2, 2026-09-18, by gateharden.
# v1 is preserved untouched at _ga_gate.sh (sha16 2c5296d492415212) and as _ga_gate.sh.v1_* if made.
#
# WHAT A GREEN VERDICT ASSERTS
#   On THIS artifact and THIS binary, the speculative backend named on the command line produced a
#   COMPLETE greedy token-id stream that is BIT-IDENTICAL to a reference stream produced by the
#   SAME recipe with speculation explicitly OFF (`--spec none`).  Every arm must have produced a
#   non-empty, full-length stream; a missing/short arm is a FAILURE, not a note.
#
# WHY v1 WAS VOID -- "three arms that are one arm"
# -----------------------------------------------
#   v1 ran four arms and compared THREE of them against the fourth:
#       run plain                                # NO --spec  -> inherits the front-end default
#       run dflash2 --spec dflash2
#       run mtp     --spec mtp --draft-tokens 3 --lm-head-draft
#       run auto    --spec auto
#   and then took `plain` as the base for the comparison.
#   The front-end default for --spec became `auto` in commit b4c95ae (2026-09-12 16:30:22), and for
#   the dflash2 artifact this gate uses, `auto` RESOLVES TO dflash2.  So on the dflash2 artifact:
#       plain == auto == dflash2 == <the backend under test>
#   Three of the four arms were the SAME arm, and the reference leg WAS the leg under test.  The
#   comparison therefore degenerated to dflash2-vs-dflash2: it could not fail, and the word
#   "consistent" in its output asserted nothing about dflash2 at all.
#   (The same inheritance makes v1's output unusable for its own secondary purpose -- reporting what
#    `auto` resolves to -- because the arm it reports as "plain" is that same resolution.)
#
# WHAT v2 CHANGES
#   1. The reference arm is ALWAYS an explicit `--spec none`.  Never inherited, never defaulted.
#   2. The backend to gate is a REQUIRED argument; `auto` and `none` are REFUSED, because a verdict
#      that does not name what it gated is exactly how v1's green became unfalsifiable.
#   3. The `auto` arm survives as a RESOLUTION PROBE: it still reports which backend the default
#      resolves to and whether that stream matches the named backend -- useful, and it is labelled
#      as carrying NO verdict.  It can never produce a green.
#   4. The exit status is driven by the pair result and by arm completeness:
#        0 = green (every arm complete, named spec bit-identical to explicit none)
#        3 = red   (an arm failed/short, or a real DIFFER)
#        2 = refused (bad/missing/ambiguous <spec>, unknown argument)
#      v1 ALWAYS exited 0 -- its verdict never reached the exit status.
#
# ENV (for test doubles; defaults are the real thing)
#   NINFER_GA_GATE_BIN    binary to run            (default ./apps/ninfer)
#   NINFER_GA_GATE_MODEL  artifact to load         (default /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer)
#   NINFER_GA_GATE_LOGDIR where per-arm logs go    (default /mnt/c/Users/User/Documents/ziqinzhang/dl)
#   NINFER_GA_GATE_BUILD  build dir to cd into     (default /home/user/ninfer-fusion/build)
#   NINFER_GA_GATE_DRYRUN 1 = build the command lines, print them, run nothing
set -u

usage() {
  cat >&2 <<'EOF'
usage: _ga_gate2.sh <spec> [--max-new N]

  <spec>  the speculative backend to gate: one of mtp | dflash | dflash2
          'auto' is REFUSED: it names no backend, so a verdict on it would say nothing
          (it is still probed and reported, but it carries no verdict).
          'none'/'off' are REFUSED: that is the reference leg, not something to gate.

  Exit status: 0 = green, 3 = red (arm failed or real DIFFER), 2 = refused.
  Green is printed only if EVERY arm produced a full stream and the named spec is
  BIT-IDENTICAL to an explicit `--spec none` stream from the same recipe.
EOF
  exit 2
}

[ $# -ge 1 ] || usage
SPEC=""; MAXNEW=96
while [ $# -gt 0 ]; do
  case "$1" in
    --max-new) MAXNEW="${2:?--max-new needs a value}"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "REFUSED: unknown argument '$1'" >&2; usage ;;
    *)  [ -z "$SPEC" ] || { echo "REFUSED: more than one spec given ('$SPEC' and '$1')" >&2; usage; }
        SPEC="$1"; shift ;;
  esac
done
case "$SPEC" in
  ""|auto)
    echo "REFUSED: <spec> must name a concrete speculative backend." >&2
    echo "  '${SPEC:-<empty>}' does not.  This refusal is the whole point: v1's reference leg" >&2
    echo "  inherited the default (auto -> dflash2), so it compared dflash2 with itself." >&2
    usage ;;
  none|off) echo "REFUSED: '$SPEC' is the REFERENCE leg, not a backend to gate." >&2; usage ;;
  mtp|dflash|dflash2) : ;;
  *) echo "REFUSED: unknown spec '$SPEC' (want mtp|dflash|dflash2)" >&2; usage ;;
esac

R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
BUILD="${NINFER_GA_GATE_BUILD:-$R/build}"
BIN="${NINFER_GA_GATE_BIN:-./apps/ninfer}"
ART="${NINFER_GA_GATE_MODEL:-/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer}"
LOG_DIR="${NINFER_GA_GATE_LOGDIR:-$J/dl}"
DRYRUN="${NINFER_GA_GATE_DRYRUN:-0}"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG="$LOG_DIR/ga_gate_${SPEC}.log"
mkdir -p "$LOG_DIR"
: > "$LOG"
# tee: the verdict must reach the CONSOLE as well as the file.  A gate whose success is
# indistinguishable from not having run is not a gate.
exec > >(tee -a "$LOG") 2>&1

# identity of the instrument: a green is only meaningful next to these
sha16() { sha256sum < "$1" 2>/dev/null | cut -c1-16 ; }
cd "$BUILD" || { echo "G-A FAIL: no build dir $BUILD"; exit 3; }
BIN16=$(sha16 "$BIN"); ART16=$(sha16 "$ART")
echo "=================================================================="
echo "=== G-A gate v2  spec=$SPEC  $(date '+%F %H:%M:%S') ==="
echo "  build dir=$BUILD  binary=$BIN (sha16=${BIN16:-MISSING})  artifact sha16=${ART16:-MISSING}"
[ -n "$BIN16" ] || { echo "G-A FAIL: binary $BIN not found under $BUILD"; exit 3; }
[ -n "$ART16" ] || { echo "G-A FAIL: artifact $ART not found"; exit 3; }

# per-backend recipe for the under-test arm
case "$SPEC" in
  mtp)     TEST_EXTRA=(--draft-tokens 3 --lm-head-draft) ;;
  dflash)  TEST_EXTRA=(--draft-tokens 7) ;;
  dflash2) TEST_EXTRA=() ;;
esac

# one arm.  THE SPEC IS ALWAYS EXPLICIT -- never inherited from a front-end default.
one() {
  local tag="$1" spec="$2"; shift 2
  local log="$LOG_DIR/ga_${SPEC}_${tag}.log" ids="$LOG_DIR/ga_${SPEC}_${tag}.ids"
  local cmd=("$BIN" "$ART" --prompt "$P" --max-new "$MAXNEW" --max-context 4096 \
             --no-thinking --greedy --seed 7 --print-token-ids --spec "$spec" "$@")
  if [ "$DRYRUN" = 1 ]; then printf '  DRYRUN [%s] %s\n' "$tag" "${cmd[*]}"; return 0; fi
  timeout 900 "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  grep -iE 'generated ids' "$log" | tail -1 | sed -E 's/.*[Gg]enerated ids[[:space:]]*//' \
      | tr -s ' \t' '\n' | grep -E '^-?[0-9]+$' > "$ids"
  local n; n=$(wc -l < "$ids")
  echo "  [$tag spec=$spec] rc=$rc n=$n"
  grep -m 1 -E 'mtp acceptance rate|dflash2 acceptance rate|dflash acceptance rate' "$log" | sed 's/^/    /'
  grep -m 1 'decode speed' "$log" | sed 's/^/    /'
  grep -m 1 -iE 'auto.*(dflash2|dflash|mtp)|resolved' "$log" | sed 's/^/    AUTO: /'
  [ "$rc" = 0 ] || { echo "    ARM-FAILED rc=$rc"; return 1; }
  [ "$n" -ge "$MAXNEW" ] || { echo "    ARM-SHORT n=$n (wanted >=$MAXNEW)"; return 1; }
  return 0
}

echo "--- arms: reference is ALWAYS explicit '--spec none' ---"
nfail=0
one ref none                  || nfail=$((nfail+1))
one tst "$SPEC" "${TEST_EXTRA[@]}" || nfail=$((nfail+1))
echo "--- probe arm (NO VERDICT): what does the front-end default resolve to? ---"
one auto auto                 || echo "    (auto probe unavailable; carries no verdict)"

if [ "$DRYRUN" = 1 ]; then echo "DRYRUN complete"; exit 0; fi

echo "--- bit comparison: explicit none  vs  $SPEC ---"
SPEC="$SPEC" MAXNEW="$MAXNEW" LOG_DIR="$LOG_DIR" python3 - <<'PY'
import os, pathlib, sys
spec = os.environ["SPEC"]; d = pathlib.Path(os.environ["LOG_DIR"])
def ids(tag):
    p = d / f"ga_{spec}_{tag}.ids"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
A, B = ids("ref"), ids("tst")
if not A or not B:
    print(f"  NO DATA ref={len(A)} spec={len(B)}"); print("GA_PAIR_BAD=no-data"); sys.exit(0)
n = min(len(A), len(B))
i = next((k for k in range(n) if A[k] != B[k]), n)
if i == n and len(A) == len(B):
    print(f"  IDENTICAL ({len(A)} tok)  ref(none) == spec({spec})"); print("GA_PAIR_BAD=")
else:
    lo = max(0, i - 2)
    print(f"  DIFFER at {i}/{n}  ref={A[lo:i+3]} spec={B[lo:i+3]}")
    print("GA_PAIR_BAD=" + spec)
PY
badline=$(grep -o 'GA_PAIR_BAD=.*' "$LOG" | tail -1)
echo "--- auto resolution probe (informational, carries NO verdict) ---"
SPEC="$SPEC" LOG_DIR="$LOG_DIR" python3 - <<'PY'
import os, pathlib, re
spec = os.environ["SPEC"]; d = pathlib.Path(os.environ["LOG_DIR"])
def ids(tag):
    p = d / f"ga_{spec}_{tag}.ids"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
t, a = ids("tst"), ids("auto")
txt = (d / f"ga_{spec}_auto.log")
m = None
if txt.exists():
    mm = re.search(r'auto[^\n]*(dflash2|dflash|mtp)', txt.read_text(errors="replace"), re.I)
    m = mm.group(1) if mm else None
print(f"  auto resolved to: {m or '(not reported by the binary)'}")
print(f"  auto stream vs {spec} stream: {'identical' if a and a == t else 'differ' if a and t else 'no data'}"
      f"   -> this arm NEVER contributes to the verdict")
PY

if [ "$nfail" -ne 0 ]; then
  echo "G-A FAIL: $nfail arm(s) did not produce a complete stream -- green is not reachable"
  echo "GA_GATE_RC=3"; exit 3
fi
case "$badline" in
  GA_PAIR_BAD=)
    echo "G-A ok  spec=$SPEC vs explicit none | artifact=$ART16 | binary=$BIN16 | $MAXNEW tok | $(date '+%F %T')"
    echo "         (v1 would have printed this same word while comparing dflash2 with dflash2)"
    echo "GA_GATE_RC=0"; exit 0 ;;
  *)
    echo "G-A FAIL: ${badline#GA_PAIR_BAD=} (see the DIFFER line above)"
    echo "GA_GATE_RC=3"; exit 3 ;;
esac
