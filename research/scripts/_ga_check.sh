#!/bin/bash
# _ga_check.sh -- G-A gate.  v3, 2026-09-18.
#
# v3 DOES NOT CHANGE WHAT A GREEN MEANS -- see the v2 block below for that contract.  v3 fixes
# two ways a NON-green could still present as green, and one way a green could present as
# nothing at all:
#   (A) v2 put BOTH streams into the log (v2 L85) so on the console success read as silence.
#       v3 keeps the console (fd 3/4) and announces the verdict and the exit status through an
#       EXIT trap, so "it ran" is never ambiguous.
#   (B) v2 read its verdict with `grep -o 'GA_PAIRS_BAD=.*' "$LOG" | tail -1` over an APPENDED
#       log.  If python died before printing, the gate inherited the PREVIOUS run's line -- and
#       a previous green made it print `G-A ok` and exit 0 without comparing anything.
#       v3 reads a per-run sentinel file, and a missing/short/crashed sentinel is a FAILURE.
#   (C) a python crash is now written to the sentinel as COMPARISON-CRASHED, never as green.
#
# WHAT A GREEN VERDICT ASSERTS
# ---------------------------
#   A green verdict (`G-A ok`) asserts exactly this, and nothing more:
#   on THIS artifact and THIS binary, for each prompt named in the verdict, the speculative path
#   named by the REQUIRED command-line argument (<spec>) produced a COMPLETE greedy token-id
#   stream that is BIT-IDENTICAL to a reference stream produced by the SAME invocation recipe with
#   speculation explicitly OFF (`--spec none`).  Every arm must have produced a non-empty stream;
#   a missing/short/failed arm is a FAILURE, not a note.  The verdict line is stamped with the
#   spec, the artifact sha16, the binary sha16, the prompt tags, and the token count, so a reader
#   can tell WHAT was gated, on WHICH build.
#
# WHY v1 WAS VOID
# ---------------
#   v1 (sha16 0513f7f54963c603) had two independent holes:
#   (1) its "plain"/reference arm passed NO --spec and therefore inherited the front-end default.
#       That default became `auto` in commit b4c95ae (2026-09-12 16:30:22, "cli/serve: --spec
#       defaults to auto ... add --spec none|off"), and the front ends still document it that way
#       (src/product/speculative_options.h: "both front ends default --spec to auto").  For the
#       dflash2 artifact this gate uses, `auto` RESOLVES to DFlash2 -- so the reference leg became
#       the very backend under test: the comparison degenerated to dflash2-vs-dflash2 and could no
#       longer fail.  The owner's own measurement dl/_auto_default.txt proves the resolution:
#       no flag -> "dflash2 draft window 7", 84.47% acceptance, 268.12 tok/s.
#   (2) the script exited 0 unconditionally -- the pair results never reached the exit status, so a
#       caller checking rc got green even when every pair printed DIFFER.
#   Both holes are closed below: the spec is a REQUIRED argument, `auto` is refused outright, the
#   reference arm is always an explicit `--spec none`, and the exit status is 0 iff every arm ran
#   and every pair matched.
set -u

usage() {
  cat >&2 <<'EOF'
usage: _ga_check.sh <spec> [--prompts zh,num]

  <spec>  the speculative backend to gate: one of mtp | dflash | dflash2
          'auto' is REFUSED: it names no backend, so a verdict on it would say nothing.
          'none'/'off' are REFUSED: that is the reference leg, not something to gate.

  Green is printed only if EVERY prompt's speculation stream is complete and bit-identical to an
  explicit `--spec none` stream from the same recipe.  Exit status: 0 = green, 3 = red, 2 = refused.
EOF
  exit 2
}

[ $# -ge 1 ] || usage
SPEC=""
PROMPTS="zh,num"
while [ $# -gt 0 ]; do
  case "$1" in
    --prompts) PROMPTS="${2:?--prompts needs a value}"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "REFUSED: unknown argument '$1'" >&2; usage ;;
    *)  [ -z "$SPEC" ] || { echo "REFUSED: more than one spec given ('$SPEC' and '$1')" >&2; usage; }
        SPEC="$1"; shift ;;
  esac
done

# ---- the refusal that makes a green verdict mean something --------------------------------
case "$SPEC" in
  ""|auto)
    echo "REFUSED: <spec> must name a concrete speculative backend." >&2
    echo "  '${SPEC:-<empty>}' does not.  A verdict that does not name what it gated is void," >&2
    echo "  which is exactly how v1's green verdicts became unfalsifiable." >&2
    usage ;;
  none|off)
    echo "REFUSED: '$SPEC' is the REFERENCE leg, not a backend to gate." >&2
    usage ;;
  mtp|dflash|dflash2) : ;;
  *)
    echo "REFUSED: unknown spec '$SPEC' (want mtp|dflash|dflash2)" >&2
    usage ;;
esac

R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG="${NINFER_GA_LOG:-$J/dl/ga_check.log}"
TMPD="${NINFER_GA_TMP:-/tmp}"
ART="${NINFER_GA_ARTIFACT:-/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer}"
BIN="${NINFER_GA_BIN:-./apps/ninfer}"

export PATH="/home/user/.local/bin:$PATH"
mkdir -p "$TMPD"
# v3: keep the console.  fd3 = the caller's stdout, fd4 = the caller's stderr.  Everything below
# still goes to the log, and every line that carries a verdict ALSO goes to fd3, so a caller who
# sees rc=0 has positively seen the word that the green asserts.
exec 3>&1 4>&2
exec >> "$LOG" 2>&1                      # both streams kept -- nothing is dropped on the floor
RUNID="$(date '+%Y%m%d-%H%M%S')-$$"
SENT="$TMPD/ga_pairs_bad.$SPEC.$RUNID"          # THIS run's verdict, written by THIS run's python
VERDICT=""
say()   { printf '%s\n' "$*" >>"$LOG"; printf '%s\n' "$*" >&3; }
announce() { local rc=$?; ${VERDICT:+true}
  say "G-A CONSOLE VERDICT: ${VERDICT:-<none>} | spec=$SPEC | run=$RUNID | rc=$rc"
  say "GA_CHECK_RC=$rc"; }
trap announce EXIT
echo "=================================================================="
echo "=== G-A gate   spec=$SPEC   $(date '+%F %H:%M:%S') ==="

cd "$R/build" || { echo "G-A FAIL: no build dir $R/build"; say "G-A FAIL: no build dir $R/build"; VERDICT="FAIL(no-build)"; exit 3; }

# identity of the instrument: a green is only meaningful next to these
sha16() { sha256sum < "$1" 2>/dev/null | cut -c1-16 ; }
ART16=$(sha16 "$ART"); BIN16=$(sha16 "$BIN")
echo "  artifact sha16=$ART16   binary sha16=$BIN16"
[ -n "$ART16" ] || { echo "G-A FAIL: artifact $ART missing"; say "G-A FAIL: artifact $ART missing"; VERDICT="FAIL(no-artifact)"; exit 3; }
[ -n "$BIN16" ] || { echo "G-A FAIL: binary $BIN missing"; say "G-A FAIL: binary $BIN missing"; VERDICT="FAIL(no-binary)"; exit 3; }

SHEN='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NUM='0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9'
prompt_of() { case "$1" in zh) printf '%s' "$SHEN" ;; num) printf '%s' "$NUM" ;; esac; }

# one arm. SPEC IS ALWAYS EXPLICIT -- never inherited.  Extra args are per-backend.
one() {
  local tag=$1 p=$2 spec=$3; shift 3
  local log="$TMPD/ga_${SPEC}_$tag.log"
  timeout 900 "$BIN" "$ART" --prompt "$p" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids --spec "$spec" "$@" > "$log" 2>&1
  local rc=$?
  local ids="$TMPD/ga_${SPEC}_$tag.ids"
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > "$ids"
  local n; n=$(wc -w < "$ids")
  say "  [$tag spec=$spec] rc=$rc n=$n"
  # an arm that did not produce a full stream is a FAILURE, not a note
  [ "$rc" = 0 ] || { echo "    ARM-FAILED rc=$rc"; say "    ARM-FAILED rc=$rc"; return 1; }
  [ "$n" -ge 96 ] || { echo "    ARM-SHORT n=$n (wanted >=96)"; say "    ARM-SHORT n=$n (wanted >=96)"; return 1; }
  return 0
}

# per-backend recipe for the under-test arm (the reference arm is always plain `none`)
case "$SPEC" in
  mtp)     TEST_EXTRA=(--draft-tokens 3) ;;
  dflash)  TEST_EXTRA=(--draft-tokens 7) ;;
  dflash2) TEST_EXTRA=() ;;
esac

echo "--- reference arms (--spec none, explicit) + test arms (--spec $SPEC) ---"
nfail=0
for pg in $(printf '%s' "$PROMPTS" | tr ',' ' '); do
  one "ref_$pg" "$(prompt_of "$pg")" none        || nfail=$((nfail+1))
  one "tst_$pg" "$(prompt_of "$pg")" "$SPEC" "${TEST_EXTRA[@]}" || nfail=$((nfail+1))
done

echo "--- bit comparison (reference vs spec) ---"
SPEC="$SPEC" PROMPTS="$PROMPTS" TMPD="$TMPD" SENT="$SENT" python3 - <<'PY'
import os, pathlib, sys, traceback
spec = os.environ["SPEC"]; tmpd = pathlib.Path(os.environ["TMPD"])
sent = pathlib.Path(os.environ["SENT"])
pg   = os.environ["PROMPTS"].split(",")
def write(v): sent.write_text(v + "\n")
def ids(tag):
    p = tmpd / f"ga_{spec}_{tag}.ids"
    return p.read_text().split() if p.exists() else []
bad = []
try:
    for g in pg:
        A, B = ids(f"ref_{g}"), ids(f"tst_{g}")
        if not A or not B:
            print(f"  {g}: NO DATA  (ref={len(A)} spec={len(B)} tok)"); bad.append(g); continue
        n = min(len(A), len(B))
        i = next((k for k in range(n) if A[k] != B[k]), n)
        if i == n and len(A) == len(B):
            print(f"  {g}: IDENTICAL ({len(A)} tok)  ref(none) == spec({spec})")
        else:
            print(f"  {g}: DIFFER at {i}/{n}  ref={A[max(0,i-2):i+3]} spec={B[max(0,i-2):i+3]}")
            bad.append(g)
    write("GA_PAIRS_BAD=" + ",".join(bad))
except BaseException:
    traceback.print_exc()
    write("GA_PAIRS_BAD=COMPARISON-CRASHED")   # never inherit a historic green
PY
# v3: the verdict comes from THIS run's sentinel FILE, never from a grep over the appended log.
# A sentinel that is absent, empty, or not from this run is a FAILURE, not an inheritance.
if [ -s "$SENT" ]; then badline=$(head -1 "$SENT"); else badline="MISSING-SENTINEL($SENT)"; fi
say "  comparison sentinel: $badline"

if [ "$nfail" -ne 0 ]; then
  VERDICT="FAIL(arms)"
  say "G-A FAIL: $nfail arm(s) did not produce a complete stream -- green is not reachable"
  exit 3
fi
case "$badline" in
  GA_PAIRS_BAD=)
    VERDICT="G-A ok"
    say "G-A ok  spec=$SPEC vs none | prompts=$PROMPTS | artifact=$ART16 | binary=$BIN16 | 96 tok | $(date '+%F %T')"
    say "GA_CHECK_DONE"
    exit 0 ;;
  *)
    VERDICT="G-A FAIL(${badline#GA_PAIRS_BAD=})"
    say "G-A FAIL: ${badline#GA_PAIRS_BAD=}"
    exit 3 ;;
esac
