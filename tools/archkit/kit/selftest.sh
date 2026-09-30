#!/usr/bin/env bash
# kit/selftest.sh — the whole shared kit, in one command, with no GPU and no download.
#
#   bash tools/archkit/kit/selftest.sh [<code-root> [<work-dir> [<live-root>]]]
#
# <code-root>  where this kit and the new model's converter live (the tree you run).
# <live-root>  the tree the read-only parts consult: the insertion-point anchors, the
#              existing targets the reverse diff is run against, the spec files of
#              those targets.  Defaults to <code-root>.  When the kit is being staged
#              in a sandbox, pass the live tree here -- every use of it is a read.
#
# Six steps, each printing its own reading.  The exit status is the number of failed
# steps.  Nothing under either root is written by this script.
set -u
CODE=${1:-$(cd "$(dirname "$0")/../../.." && pwd)}
WORK=${2:-/tmp/archkit-selftest}
LIVE=${3:-$CODE}
KIT="$CODE/tools/archkit/kit"
SPEC="$LIVE/tools/archkit/specs/minicpm5_1b_spec.json"

echo "RC_START=selftest"
echo "   code root $CODE"
echo "   live root $LIVE   (read-only: anchors, existing targets, their specs)"
echo "   work      $WORK"
rm -rf "$WORK"; mkdir -p "$WORK"
fails=0

step() { # step <name> <rc>
  if [ "$2" = "0" ]; then echo "STEP $1: OK"; else echo "STEP $1: FAIL (rc=$2)"; fails=$((fails+1)); fi
}

echo "== 1/6 the demo: a synthetic checkpoint through the shared driver =="
(cd "$CODE" && python3 -m tools.archkit.kit.demo --out "$WORK/demo") > "$WORK/1.log" 2>&1
step demo $?
grep -E 'sha256|magic|objects ' "$WORK/1.log" | sed 's/^/   /'

echo "== 2/6 the negative battery =="
(cd "$CODE" && python3 -m tools.archkit.kit.test_kit) > "$WORK/2.log" 2>&1
step negatives $?
grep -E '^== ' "$WORK/2.log" | sed 's/^/   /'

echo "== 3/6 the scaffold generator =="
(cd "$CODE" && python3 -m tools.archkit.kit.scaffold --spec "$SPEC" \
   --id minicpm5_1b --out "$WORK/scaffold/minicpm5_1b" --repo-root "$LIVE") \
   > "$WORK/3.log" 2>&1
step scaffold $?
grep -E 'insert:|zero-reference accessors measured' "$WORK/3.log" | sed 's/^/   /'

echo "== 4/6 the compile gate (R41 + R52 + RED + LOCK) =="
SBX="$WORK/cgate" bash "$KIT/compilegate.sh" "$WORK/scaffold/minicpm5_1b" \
  > "$WORK/4.log" 2>&1
step compilegate $?
grep -E 'ARM |RC_FAILED_ARMS' "$WORK/4.log" | sed 's/^/   /'

echo "== 5/6 the ruler on our own candidate =="
(cd "$CODE" && python3 -m tools.archkit.kit.check --check \
   "$WORK/scaffold/minicpm5_1b" --spec "$SPEC" --repo-root "$LIVE") \
   > "$WORK/5.log" 2>&1
step check $?
grep -E '^   (files|spec|TODO|places|tier)' "$WORK/5.log" | sed 's/^/   /'

echo "== 6/6 reverse validation against two targets the generator never saw =="
rc6=0
for target in muse_glimmer_30b spark_x2_5_4b; do
  # The spec files are NOT named after the target directory: `muse_glimmer_30b` has
  # `muse-glimmer-30b_spec.json` and `spark_x2_5_4b` has `spark-x2.5-4b_spec.json`
  # (a dot, from the model's own spelling).  Translating underscores to hyphens gets
  # one right and one wrong, so the lookup is by glob and the resolved name is
  # printed -- a missing spec must be visible, not silently skipped.
  case "$target" in
    muse_glimmer_30b) candidates="muse-glimmer-30b_spec.json" ;;
    spark_x2_5_4b)    candidates="spark-x2.5-4b_spec.json spark-x2-5-4b_spec.json" ;;
    *)                candidates="$(echo "$target" | tr '_' '-')_spec.json" ;;
  esac
  spec=""
  for name in $candidates; do
    if [ -f "$LIVE/tools/archkit/specs/$name" ]; then spec="$LIVE/tools/archkit/specs/$name"; break; fi
  done
  if [ -z "$spec" ]; then
    echo "   [$target] REFUSED: none of [$candidates] exists under $LIVE/tools/archkit/specs/"
    rc6=2
    continue
  fi
  echo "   [$target] spec $spec"
  (cd "$CODE" && python3 -m tools.archkit.kit.check --reverse \
     --spec "$spec" --id "$target" \
     --config "$LIVE/src/targets/$target/impl/config.h" --repo-root "$LIVE") \
     > "$WORK/6.$target.log" 2>&1
  r=$?
  [ "$r" = "0" ] || rc6=$r
  grep -E 'MECHANICAL:|HAND:|GAPS:|BREACH|verdict' "$WORK/6.$target.log" \
    | sed "s/^/   [$target] /"
done
step reverse "$rc6"

echo "RC_FAILED_STEPS=$fails"
echo "RC_END=selftest"
exit $fails
