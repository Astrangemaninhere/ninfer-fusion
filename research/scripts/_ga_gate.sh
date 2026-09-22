#!/bin/bash
# _ga_gate.sh -- G-A gate: is the speculative backend's greedy token-id stream BIT-IDENTICAL to a
# reference stream produced by the SAME recipe with speculation explicitly OFF?
#
# v1h (rev 3), 2026-09-18, line `gaatomic` -- HARDENING OF v1 (sha16 2c5296d492415212), which was VOID.
# ------------------------------------------------------------------------------------------------
# WHAT WAS WRONG WITH v1.  Every defect below is already named in the header of the sibling
# _ga_check.sh v3 and in dl/gateharden/REPORT.md §5 / §6.2; this file closes them BY CONSTRUCTION,
# so that no future caller can reach the old behaviour by re-reading the old file.
#   (1) INHERITED REFERENCE LEG.  v1's `plain` arm passed NO --spec, so it inherited the front-end
#       default.  That default became `auto` in commit b4c95ae (2026-09-12 16:30:22), and for the
#       dflash2 artifact this gate loads, `auto` RESOLVES TO dflash2.  So plain == auto == dflash2
#       == THE BACKEND UNDER TEST: three of the four arms were the same arm, the comparison
#       degenerated to dflash2-vs-dflash2, and it could not fail.  The word "一致" in v1's output
#       asserted nothing about dflash2 at all.
#   (2) NO EXIT STATUS.  v1's last line was `echo GA_GATE_DONE`; the pair results never reached the
#       exit status.  A caller checking rc got 0 even when every pair had printed a DIFFER.
#   (3) EVIDENCE DESTRUCTION.  v1 truncated $J/dl/ga_gate.log and overwrote the four per-arm logs
#       $J/dl/ga_{plain,dflash2,mtp,auto}.log on EVERY run -- the exact paths the open `29/96`
#       investigation reads (all five were absent as of 2026-09-18 22:49).  Gone: v1h writes
#       nothing there, defaults its logs to /tmp, stamps every file with a per-run RUNID, and
#       refuses LOG_DIR == that one directory.
# v1h closes all three: the backend is a REQUIRED argument, `auto` is refused outright, the reference
# arm is always an explicit `--spec none`, and the exit status is DERIVED from arm completeness plus
# the pair result (not declared).
#
# REVISIONS, each from a measurement on a test double (no GPU was used at any point):
#   rev 2  rev 1 refused any LOG_DIR *under* $J/dl, and the first end-to-end proof hit that refusal
#          on a harmless sandbox of its own (dl/gaatomic/_double/logs -> rc=2).  A guard that fires
#          on correct configurations trains callers to route around it, so the refusal is narrowed
#          to exactly the one directory that holds the five evidence paths, and the no-argument path
#          now STATES the refusal before its usage text.
#   rev 3  the comparison read the per-arm .ids file LINE BY LINE while the arm wrote all ids on ONE
#          line, so a real divergence at index 29 was reported as "DIFFER at 0/1" -- the verdict was
#          still right, the COORDINATES were garbage, and index coordinates are the whole currency of
#          the `29/96` investigation.  The ids are now split on whitespace, exactly as _ga_check.sh
#          v3 does it (`p.read_text().split()`).
#
# WHAT A GREEN ASSERTS (GA_GATE_RC=0), and nothing more: on THIS binary (sha16 stamped below) and on
# THIS artifact (sha16 stamped below), the speculation backend NAMED ON THE COMMAND LINE produced a
# COMPLETE greedy token-id stream that is bit-identical to a reference stream from the SAME recipe
# with `--spec none` given EXPLICITLY.  A missing or short arm is a FAILURE, not a note.
#
# EXIT STATUS (derived): 0 = green | 3 = red (an arm was incomplete, or a real DIFFER)
#                        2 = refused (missing / unnamed / unknown <spec>, or LOG_DIR == $J/dl)
#
# usage: _ga_gate.sh <spec> [--max-new N]        <spec> must be one of mtp | dflash | dflash2
#
# ENV -- the defaults are the real thing; the NINFER_GA_GATE_* names are shared with _ga_gate2.sh so
# that a test double can be substituted through them.  That is how this gate was proven with no GPU.
#   NINFER_GA_GATE_BIN      binary to run          (default ./apps/ninfer)
#   NINFER_GA_GATE_MODEL    artifact to load       (default /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer)
#   NINFER_GA_GATE_LOGDIR   where per-arm logs go  (default /tmp/ga_gate_v1h -- NEVER the dl/ area)
#   NINFER_GA_GATE_BUILD    build dir to cd into   (default /home/user/ninfer-fusion/build)
set -u

usage() {
  cat >&2 <<'EOF'
usage: _ga_gate.sh <spec> [--max-new N]

  <spec>  the speculative backend to gate: one of mtp | dflash | dflash2
          'auto' is REFUSED: it names no backend, so a verdict on it would say nothing.  It is
          still probed and reported (v1's one useful output), but it carries NO verdict.
          'none'/'off' are REFUSED: that is the REFERENCE leg, not something to gate.

  Exit status: 0 = green, 3 = red (an arm was incomplete or a real DIFFER), 2 = refused.
  Green is printed only if EVERY arm produced a full stream AND the named spec is BIT-IDENTICAL
  to an explicit `--spec none` stream from the same recipe.
EOF
  exit 2
}

[ $# -ge 1 ] || {
  echo "REFUSED: <spec> must name a concrete speculative backend." >&2
  echo "  no <spec> was given at all, and a verdict that does not name what it gated is void." >&2
  usage
}
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
    echo "  inherited the front-end default (auto -> dflash2), so it compared dflash2 with itself." >&2
    usage ;;
  none|off) echo "REFUSED: '$SPEC' is the REFERENCE leg, not a backend to gate." >&2; usage ;;
  mtp|dflash|dflash2) : ;;
  *) echo "REFUSED: unknown spec '$SPEC' (want mtp|dflash|dflash2)" >&2; usage ;;
esac

R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
BUILD="${NINFER_GA_GATE_BUILD:-$R/build}"
BIN="${NINFER_GA_GATE_BIN:-./apps/ninfer}"
ART="${NINFER_GA_GATE_MODEL:-/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer}"
LOG_DIR="${NINFER_GA_GATE_LOGDIR:-/tmp/ga_gate_v1h}"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

# The five arm-log names the 29/96 investigation reads live DIRECTLY in $J/dl, and v1 truncated and
# overwrote them on every run.  v1h can no longer produce those names -- it defaults to /tmp and
# stamps every file with a per-run RUNID -- and this refuses the one directory that holds them.
# Subdirectories are deliberately NOT refused, so a line may still sandbox its own logs under dl/.
case "$LOG_DIR" in
  "$J/dl")
    echo "REFUSED: LOG_DIR=$LOG_DIR is the directory that holds the G-A arm logs." >&2
    echo "  The open 29/96 investigation reads them, and v1 truncated and overwrote them on" >&2
    echo "  every run.  Point LOG_DIR at a subdirectory of your own, or leave the default." >&2
    exit 2 ;;
esac

RUNID="$(date '+%Y%m%d-%H%M%S')-$$"
mkdir -p "$LOG_DIR" || { echo "G-A FAIL: cannot create LOG_DIR=$LOG_DIR"; echo "GA_GATE_RC=3"; exit 3; }
LOG="$LOG_DIR/ga_gate.$RUNID.log"
: > "$LOG"
# tee: the verdict must reach the CONSOLE as well as the file.  Silence-on-success was a second way
# for v1 to be unfalsifiable -- a gate whose success looks like not having run is not a gate.
exec > >(tee -a "$LOG") 2>&1
sha16() { sha256sum < "$1" 2>/dev/null | cut -c1-16; }

cd "$BUILD" || { echo "G-A FAIL: no build dir $BUILD"; echo "GA_GATE_RC=3"; exit 3; }
BIN16=$(sha16 "$BIN"); ART16=$(sha16 "$ART")
echo "=================================================================="
echo "=== G-A gate v1h  spec=$SPEC  $(date '+%F %H:%M:%S') ==="
echo "  build dir=$BUILD  binary=$BIN (sha16=${BIN16:-MISSING})  artifact sha16=${ART16:-MISSING}"
echo "  per-arm logs: $LOG_DIR   (v1 wrote these into the dl/ evidence area; v1h does not)"
[ -n "$BIN16" ] || { echo "G-A FAIL: binary $BIN not found under $BUILD"; echo "GA_GATE_RC=3"; exit 3; }
[ -n "$ART16" ] || { echo "G-A FAIL: artifact $ART not found"; echo "GA_GATE_RC=3"; exit 3; }

# per-backend recipe for the arm under test
case "$SPEC" in
  mtp)     TEST_EXTRA=(--draft-tokens 3 --lm-head-draft) ;;
  dflash)  TEST_EXTRA=(--draft-tokens 7) ;;
  dflash2) TEST_EXTRA=() ;;
esac

# one arm.  THE SPEC IS ALWAYS EXPLICIT -- never inherited from a front-end default.
# The .ids file holds every id on ONE line (same shape _ga_check.sh v3 consumes with wc -w and
# .split()); the comparator below must therefore split on WHITESPACE, not on lines.
one() {
  local tag="$1" spec="$2"; shift 2
  local log="$LOG_DIR/ga_${SPEC}_$tag.$RUNID.log" ids="$LOG_DIR/ga_${SPEC}_$tag.$RUNID.ids"
  timeout 900 "$BIN" "$ART" --prompt "$P" --max-new "$MAXNEW" --max-context 4096 \
      --no-thinking --greedy --seed 7 --print-token-ids --spec "$spec" "$@" > "$log" 2>&1
  local rc=$?
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 \
      | sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > "$ids"
  local n; n=$(wc -w < "$ids")
  echo "  [$tag spec=$spec] rc=$rc n=$n"
  grep -m 1 -E 'mtp acceptance rate|dflash2 acceptance rate|dflash acceptance rate' "$log" | sed 's/^/    /'
  grep -m 1 'decode speed' "$log" | sed 's/^/    /'
  grep -m 1 -iE 'auto.*(dflash2|dflash|mtp)|resolved' "$log" | sed 's/^/    AUTO: /'
  if [ "$rc" != 0 ];          then echo "    ARM-FAILED rc=$rc"; return 1; fi
  if [ "$n" -lt "$MAXNEW" ];  then echo "    ARM-SHORT n=$n (wanted >=$MAXNEW)"; return 1; fi
  return 0
}

echo "--- arms: the reference is ALWAYS an explicit '--spec none' (v1 inherited it) ---"
nfail=0
one ref none                        || nfail=$((nfail+1))
one tst "$SPEC" "${TEST_EXTRA[@]}"  || nfail=$((nfail+1))
echo "--- probe arm (NO VERDICT): what does the front-end default resolve to? ---"
one auto auto                       || echo "    (auto probe unavailable; carries no verdict)"

# The verdict is read from a PER-RUN sentinel written synchronously by THIS run's python, never from
# the appended log: reading an appended log can inherit a previous run's line, and v1h must not be
# able to print a green that a previous run earned.  A missing/empty sentinel is a FAILURE.
SENT="$LOG_DIR/ga_pairs_bad.$SPEC.$RUNID"
echo "--- bit comparison: explicit none  vs  $SPEC ---"
SPEC="$SPEC" MAXNEW="$MAXNEW" LOG_DIR="$LOG_DIR" RUNID="$RUNID" python3 - <<'PY' | tee "$SENT"
import os, pathlib, sys
spec = os.environ["SPEC"]; d = pathlib.Path(os.environ["LOG_DIR"]); run = os.environ["RUNID"]
def ids(tag):
    # ONE line of ids (see `one()`), so split on WHITESPACE -- exactly as _ga_check.sh v3 does.
    # A line-based read here compares whole lines and reports "DIFFER at 0/1" for a divergence at
    # index 29 (measured on the double, rev 2).  Index coordinates are the currency of the 29/96
    # investigation, so they must be right.
    p = d / f"ga_{spec}_{tag}.{run}.ids"
    return p.read_text().split() if p.exists() else []
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
prc=${PIPESTATUS[0]}

if [ "$prc" != 0 ] || [ ! -s "$SENT" ]; then
  echo "G-A FAIL: the comparison produced no result of its own (python rc=$prc) -- green is NOT reachable"
  echo "GA_GATE_RC=3"; exit 3
fi
badline=$(grep -o 'GA_PAIR_BAD=.*' "$SENT" | tail -1)
if [ "$nfail" -ne 0 ]; then
  echo "G-A FAIL: $nfail arm(s) did not produce a complete stream -- green is not reachable"
  echo "GA_GATE_RC=3"; exit 3
fi
case "$badline" in
  GA_PAIR_BAD=)
    echo "G-A ok  spec=$SPEC vs explicit none | artifact=$ART16 | binary=$BIN16 | $MAXNEW tok | $(date '+%F %T')"
    echo "         (v1 printed a green word in this same situation by comparing dflash2 with dflash2)"
    echo "GA_GATE_RC=0"; exit 0 ;;
  *)
    echo "G-A FAIL: ${badline#GA_PAIR_BAD=} (see the DIFFER line above)"
    echo "GA_GATE_RC=3"; exit 3 ;;
esac
