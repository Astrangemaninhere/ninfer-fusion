#!/bin/bash
# kv_budget_regen.sh -- regenerate the "request -> achieved -> spec" ladder of
# docs/maintainer/kv-strategy-matrix.md section 3, and CROSS-CHECK it while doing so.
#
#   bash tools/archkit/kv_budget_regen.sh                    # the doc's default budget list
#   bash tools/archkit/kv_budget_regen.sh --bits 4 4.5 6 8
#   bash tools/archkit/kv_budget_regen.sh --cold-cap 8       # the cold pool armed
#   bash tools/archkit/kv_budget_regen.sh --out /tmp/ladder.txt --keep
#
# WHY THIS EXISTS
#   The shipped section-3 table is the stdout of tools/archkit/kv_bit_budget.py, whose own
#   header banner says its ladder has forked from the authority and its stdout must not be
#   used for decisions. Every row of the shipped table reproduces that tool exactly. The
#   table also names four generators in its section 5 that are no longer on disk. This
#   script replaces the ladder's generator with a host-only probe of the ENGINE'S OWN
#   selector, so a row can no longer be a printout of a tool the engine does not run.
#
# NO GPU, NO LOCK, NO MODEL
#   src/product/kv_bit_budget.h and src/product/kv_kv_bits.h are documented host-only,
#   std-only, no CUDA (kv_kv_bits.h:5-8), so this runs at any time, in parallel with a GPU
#   arm, and its per-budget rows are written to disk rather than being held in one long
#   call's stdout.
#
# CHECKS (each failure prints a DEFECT| line and fails the run)
#   C1  the offline mirror's ladder IS the engine's ladder (read from the probe's own dump)
#   C2  --kv-bits row == --kv-bit-budget row (two spellings, one allocator)
#   C3  achieved == the plain per-layer average of the spec line, recomputed from the ladder
#   C4  the request is respected: achieved <= request
#   C5  vs the offline mirror: at cold_cap>0 identical (same DP); at cold_cap==0 a spec
#       difference is expected, but a strictly higher engine penalty, or a refusal where
#       the mirror fits, is a defect
set -u -o pipefail

TREE=${TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAGE=${STAGE:-${TMPDIR:-/tmp}/kv_budget_regen}
BITS=(3.5 4 4.5 5 6 8 12 16)
LAYERS=16
COLD_CAP=0
CC_BIN=${CC:-g++}
CODEC=iso4e
OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --bits)     shift; BITS=(); while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do BITS+=("$1"); shift; done ;;
    --layers)   LAYERS="$2"; shift 2 ;;
    --cold-cap) COLD_CAP="$2"; shift 2 ;;
    --v-codec)  CODEC="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    --stage)    STAGE="$2"; shift 2 ;;
    -h|--help)  sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -n "$OUT" ]; then exec > >(tee "$OUT") 2>&1; fi

mkdir -p "$STAGE"
PROBE_SRC="$TREE/tools/archkit/kv_budget_probe.cpp"
MIRROR_SRC="$TREE/tools/archkit/kv_budget_mirror.py"
for f in "$PROBE_SRC" "$MIRROR_SRC"; do
  [ -f "$f" ] || { echo "MISSING required file: $f" >&2; exit 2; }
done

DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HEAD=$(git -C "$TREE" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
sha16() { sha256sum "$1" | cut -c1-16; }

echo "================================================================================"
echo "KV bit-budget ladder regeneration"
echo "  date_utc        $DATE_UTC"
echo "  tree            $TREE"
echo "  git_head        $HEAD"
echo "  header sha256   src/product/kv_bit_budget.h $(sha256sum "$TREE/src/product/kv_bit_budget.h" | cut -d' ' -f1)"
echo "  header sha256   src/product/kv_kv_bits.h    $(sha256sum "$TREE/src/product/kv_kv_bits.h" | cut -d' ' -f1)"
echo "  probe           kv_budget_probe.cpp         sha16=$(sha16 "$PROBE_SRC")"
echo "  mirror          kv_budget_mirror.py         sha16=$(sha16 "$MIRROR_SRC")"
echo "  engine binaries FOR THE RECORD ONLY (this script does not run them, but a measured"
echo "                  column would be tied to one, so they belong next to these rows):"
for b in "$TREE/build/apps/ninfer" "$TREE/build/apps/ninfer-serve"; do
  if [ -f "$b" ]; then
    echo "                  $(basename "$b")  sha16=$(sha16 "$b")  mtime=$(date -u -r "$b" +%Y-%m-%dT%H:%M:%SZ)"
  fi
done
echo "  engine binary   (NOT used by this script -- the offline columns come from the engine's"
echo "                   own headers compiled host-only, so they are not tied to a build)"
echo "  criterion       THE OFFLINE SOLVER's objective is selected by --criterion:"
echo "                    lexicographic (DEFAULT) = the engine's DECLARED order as of this run,"
echo "                      'MORE BITS first, penalty only on an equal-bit tie'"
echo "                      (src/product/kv_bit_budget.h, the BUDGETSAT saturation block)"
echo "                    penalty-min             = 'min sum(penalty_x100) s.t."
echo "                      sum(bits_x100) <= round(bits*100)*L', i.e. what this doc's section 3"
echo "                      used to state; still run below for contrast"
echo "                  The ENGINE's own objective is not a parameter here -- it is whatever the"
echo "                  pinned header above implements, and the adjudicator's failure mode is"
echo "                  exactly a header that changed its objective under the table."
echo "  layers          $LAYERS   rk4v4_limit 8   cold_cap $COLD_CAP   v_codec $CODEC"
echo "  request bits    ${BITS[*]}"
echo "  NOTE: these columns are COMPUTED, not measured. The tok/s and needle-count columns"
echo "        of the matrix are a separate GPU-bound measurement and are NOT produced here."
echo "================================================================================"

echo "--- building the host probe (no CUDA, no model) ---"
# PIN the two engine headers this run is measured against. They are a documented moving
# target (the allocator header was rewritten under this session mid-run once already), so
# the run compiles against a COPY whose sha256 is printed above and below. If those two
# shas differ, the tree moved during the run and the rows must NOT be pooled with a later
# run's rows.
mkdir -p "$STAGE/pin/product"
cp "$TREE/src/product/kv_bit_budget.h" "$STAGE/pin/product/kv_bit_budget.h"
cp "$TREE/src/product/kv_kv_bits.h"    "$STAGE/pin/product/kv_kv_bits.h"
if ! "$CC_BIN" -std=c++20 -O1 -I "$STAGE/pin" -I "$TREE/include" -I "$TREE/src" \
     "$PROBE_SRC" -o "$STAGE/kbp" 2>"$STAGE/build.err"; then
  echo "BUILD FAILED:"; cat "$STAGE/build.err"; exit 3
fi
echo "  ok -> $STAGE/kbp (pinned headers in $STAGE/pin)"
echo "  header sha256 AFTER pin  kv_bit_budget.h $(sha256sum "$STAGE/pin/product/kv_bit_budget.h" | cut -d' ' -f1)"
echo "  header sha256 AFTER pin  kv_kv_bits.h    $(sha256sum "$STAGE/pin/product/kv_kv_bits.h" | cut -d' ' -f1)"
echo "  header sha256 IN TREE now kv_bit_budget.h $(sha256sum "$TREE/src/product/kv_bit_budget.h" | cut -d' ' -f1)"
echo "  header sha256 IN TREE now kv_kv_bits.h    $(sha256sum "$TREE/src/product/kv_kv_bits.h" | cut -d' ' -f1)"

echo "--- engine ladder (the authority, dumped by the probe) ---"
"$STAGE/kbp" ladder --layers "$LAYERS" --cold-cap "$COLD_CAP" | tee "$STAGE/ladder.txt"

echo "--- engine rows (both spellings) ---"
"$STAGE/kbp" rows "${BITS[@]}" --layers "$LAYERS" --cold-cap "$COLD_CAP" --v-codec "$CODEC" \
  > "$STAGE/rows_engine.txt" 2>&1
if ! grep -q '^ROW|' "$STAGE/rows_engine.txt"; then
  echo "WARN: the probe produced no rows"; cat "$STAGE/rows_engine.txt"
fi

echo "--- offline mirror (the engine's DECLARED criterion) ---"
# F_l: the engine's per-layer admissible gear set, read from the engine's OWN refusal text
# ("candidate space at layer 0: bf16,int8,nvfp4,rk4v4,iso4e,rk3v4,rk2v4") rather than re-derived.
# Without it the mirror keeps beating the engine with fp8, which the gearbox drops under the
# default component mode, and would report the engine's own constraint as a defect.
GEARS=$(grep -o 'candidate space at layer 0: [a-z0-9,]*' "$STAGE/rows_engine.txt" \
        | head -1 | sed 's/.*: //' | tr -d '\r')
if [ -n "$GEARS" ]; then
  echo "  F_l from the engine's refusal text: $GEARS"
else
  echo "  F_l not named by any refusal at this budget list; mirror uses every selectable row"
fi
if ! python3 "$MIRROR_SRC" --ladder-file "$STAGE/ladder.txt" --layers "$LAYERS" \
        --bits "${BITS[@]}" --cold-cap "$COLD_CAP" --criterion lexicographic \
        ${GEARS:+--gears "$GEARS"} > "$STAGE/rows_mirror.txt" 2>&1; then
  echo "MIRROR FAILED:"; cat "$STAGE/rows_mirror.txt"; exit 4
fi
grep '^MIRROR|' "$STAGE/rows_mirror.txt" | sed 's/^/  /'

echo "--- cross-check + adjudication ---"
python3 "$MIRROR_SRC" --ladder-file "$STAGE/ladder.txt" --engine "$STAGE/rows_engine.txt" \
        --mirror-rows "$STAGE/rows_mirror.txt" --layers "$LAYERS" --cold-cap "$COLD_CAP" \
        --criterion lexicographic ${GEARS:+--gears "$GEARS"}
rc=$?

echo "--- the SAME budgets under the OLD criterion, for the trade it makes ---"
echo "  min-penalty-subject-to-the-ceiling is what the SHIPPED table's section 3 states and"
echo "  what every build before 2026-09-18T14:21:55Z did. It is shown so the two objectives"
echo "  are not confused: it undershoots the ceiling by design, which is the behaviour"
echo "  BUDGETSAT's saturation rule replaced."
python3 "$MIRROR_SRC" --ladder-file "$STAGE/ladder.txt" --layers "$LAYERS" \
        --bits "${BITS[@]}" --cold-cap "$COLD_CAP" --criterion penalty-min \
        ${GEARS:+--gears "$GEARS"} 2>&1 \
  | grep '^MIRROR|' | sed 's/^/  PENALTY-MIN| /'

echo "--- the FORKED tool, for contrast (NOT an authority; see its own banner) ---"
echo "  The shipped section-3 table reproduces this tool row for row. Its ladder is"
echo "  fp8 803 / rk4v4 406 / iso4e 300 against the engine's 850 / 425 / 450, and its"
echo "  iso4e penalty is 50 against the engine's pinned 200, so its rows describe a"
echo "  ladder no engine in this tree has. Divergence below is therefore expected and"
echo "  is the reason the shipped table is stale -- it is not a second opinion."
python3 "$TREE/tools/archkit/kv_bit_budget.py" --layers "$LAYERS" --bits "${BITS[@]}" 2>&1 \
  | sed 's/^/  FORKED| /'

echo "--- artefacts ---"
echo "  stage dir: $STAGE"
ls -la "$STAGE" | sed 's/^/  /'
exit $rc
