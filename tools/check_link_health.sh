#!/bin/bash
# check_link_health.sh -- catch the failure shapes a shared build directory hides.
#
# WHY THIS EXISTS
# On 2026-09-18 an archive member -- libninfer_ops.a's ops/linear/qpn/qpn_arch_route.cpp.o --
# undefined-referenced ninfer::artifact::format_name, defined only in libninfer_artifact.a, while
# ninfer_ops's link list named ninfer_core and ninfer_nvfp4_tma but not ninfer_artifact. Every
# final link that pulled that member failed, and GNU make DELETES the output of a failed link. So
# the symptom was an ABSENT BINARY, which reads as "not built", and a STALE binary hid the defect
# entirely until something forced the relink. Thirteen linear test targets were in that state.
# Attribution: scratch/PATCHSET/LINKFIX/REPORT.md. Verbatim diagnostic:
# scratch/PATCHSET/PROBE2ROUTE/arm_A1b_relink.log.
#
# CHECKS, all read-only, none of them builds anything:
#   1 ARCHIVE INTERFACE. Every final link line carrying a known archive must also carry the
#     archives that archive's interface requires (table below). THIS is the check that fails
#     BEFORE the first bad link instead of after it, and it is the one that catches this defect
#     class. The table is the only thing that has to be maintained by hand.
#   2 LINK OUTPUT MISSING THOUGH THE TARGET WAS COMPILED. Objects present, executable absent: that
#     is the exact signature make leaves behind when a link fails, and it is distinguishable from
#     "never built" (no objects either), which is a normal mid-flight state and only INFO here.
#   3 REGISTRATION. Each registration is resolved to the executable it actually RUNS (an
#     add_test name is not a target name, and a script test's artefact is a source file). Two
#     failures and they are distinguished: a name with no target and no script can NEVER be
#     built, and a name whose target exists but whose binary is absent is REGISTERED BUT
#     UNBUILT -- 'ctest -N' counts it among the total and it can only ever report '***Not Run'.
#     Both are FINDINGS by default; --allow-unbuilt downgrades only the second to a loud WARN
#     that keeps the count and every name, so accepting a mid-flight build dir is a visible act
#     rather than a silent omission.
#   4 STALENESS (summary; `--list-stale` to enumerate). A link output older than an archive on its
#     own link line will be relinked on the next build -- i.e. a place where a link defect is
#     currently INVISIBLE. Reported, never fatal: staleness is a state, not a defect. It is a
#     summary by default because a shared build dir has hundreds of them and the signal is the two
#     or three that are NOT stale.
#
# USAGE  tools/check_link_health.sh [build-dir] [--list-stale] [--allow-unbuilt]
# EXIT   0 clean (warnings allowed), 1 findings, 2 cannot measure
set -u

TREE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=""
LIST_STALE=0
ALLOW_UNBUILT=0
for a in "$@"; do
  case "$a" in
    --list-stale) LIST_STALE=1 ;;
    --allow-unbuilt) ALLOW_UNBUILT=1 ;;
    *) BUILD="$a" ;;
  esac
done
BUILD="${BUILD:-$TREE/build}"

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  echo "cannot measure: $BUILD has no CMakeCache.txt (not a configured build dir)" >&2
  exit 2
fi

# archive -> archives that must appear on any final link line carrying it.
# Extend when an archive's interface starts requiring another one.
# Both arguments are normalised (no lib prefix, no .a suffix) before the lookup.
requires() {
  case "$1" in
    ninfer_ops)      echo "ninfer_artifact" ;;
    ninfer_engine)   echo "ninfer_artifact" ;;
    ninfer_artifact) echo "ninfer_core" ;;
    ninfer_serve)    echo "ninfer_engine" ;;
    *)               echo "" ;;
  esac
}
# NOTE the trailing newline: without it every name concatenates into one token, the loop below
# sees a single un-matchable word, and THIS WHOLE CHECK SILENTLY PASSES ON EVERY INPUT. That bug
# was found by the negative control (remove ninfer_artifact from a victim's link.txt and require a
# FAIL) -- not by reading the script. Keep the control.
norm() { printf '%s\n' "$1" | sed -e 's#.*/##' -e 's/^lib//' -e 's/\.a$//'; }

fail=0; warn=0
n1=0; n2=0; n3=0; n4=0; ninfo=0; nregdir=0

echo "== link health: $BUILD =="

# ------------------------------------------------------------------ 1 archive interface
for d in "$BUILD"/*/CMakeFiles/*.dir; do
  [ -d "$d" ] || continue
  lt="$d/link.txt"
  [ -f "$lt" ] || continue
  first="$(head -c 200 "$lt" | awk '{print $1}')"
  case "$(basename "$first")" in ar|ranlib) continue ;; esac   # archive CREATION commands
  have="$(tr ' ' '\n' < "$lt" | grep -E '\.a$' | while read -r t; do norm "$t"; done | sort -u)"
  b="$(basename "$d" .dir)"
  for h in $have; do
    for need in $(requires "$h"); do
      if ! printf '%s\n' "$have" | grep -Fxq "$need"; then
        echo "FAIL  1 $b: link line carries lib$h.a but not lib$need.a ($h's interface requires it)"
        n1=$((n1+1)); fail=1
      fi
    done
  done
done
[ "$n1" -eq 0 ] && echo "ok    1 archive interface: every link line carrying a known archive carries its requirements"

# ------------------------------------------- 2 outputs, 3 registration, 4 staleness
have_ctest=0
command -v ctest >/dev/null 2>&1 && have_ctest=1
if [ "$have_ctest" = 1 ]; then
  ( cd "$BUILD" && ctest -N 2>/dev/null | sed -n 's/^  Test *#[0-9]*: *//p' | sort ) > /tmp/clh_reg.$$ || true
fi
: > /tmp/clh_dirs.$$
: > /tmp/clh_stale.$$

for d in "$BUILD"/tests/CMakeFiles/*.dir; do
  [ -d "$d" ] || continue
  name="$(basename "$d" .dir)"
  echo "$name" >> /tmp/clh_dirs.$$
  lt="$d/link.txt"
  [ -f "$lt" ] || continue
  first="$(head -c 200 "$lt" | awk '{print $1}')"
  case "$(basename "$first")" in ar|ranlib) continue ;; esac
  objs=$(ls "$d"/*.o "$d"/*/*.o "$d"/*/*/*.o 2>/dev/null | grep -vc 'cmake_device_link\.o$')
  cwd="$(dirname "$(dirname "$(dirname "$lt")")")"
  out=$(tr ' ' '\n' < "$lt" | awk '/^-o$/{getline; print; exit}')
  [ -n "$out" ] || continue
  case "$out" in /*) outpath="$out" ;; *) outpath="$cwd/$out" ;; esac
  if [ ! -e "$outpath" ]; then
    if [ "$objs" -gt 0 ]; then
      echo "FAIL  2 $name: $objs object(s) present but link output '$out' is MISSING"
      echo "         <- signature of a link that failed (make deletes the output)"
      n2=$((n2+1)); fail=1
    else
      ninfo=$((ninfo+1))
    fi
    continue
  fi
  newest=""
  for a in $(tr ' ' '\n' < "$lt" | grep -E '\.a$'); do
    case "$a" in /*) ap="$a" ;; *) ap="$cwd/$a" ;; esac
    [ -e "$ap" ] || continue
    { [ -z "$newest" ] || [ "$ap" -nt "$newest" ]; } && newest="$ap"
  done
  if [ -n "$newest" ] && [ "$newest" -nt "$outpath" ]; then
    printf '%s (newer: %s)\n' "$name" "$(basename "$newest")" >> /tmp/clh_stale.$$
  fi
done
n4=$(wc -l < /tmp/clh_stale.$$)
[ "$n4" -gt 0 ] && warn=1
if [ "$LIST_STALE" = 1 ] && [ "$n4" -gt 0 ]; then
  while IFS= read -r l; do echo "WARN  4 stale: $l"; done < /tmp/clh_stale.$$
else
  [ "$n4" -gt 0 ] && echo "warn  4 staleness: $n4 link output(s) older than an archive on their own line (use --list-stale)"
fi

  # WHAT EACH REGISTRATION ACTUALLY RUNS. The registered NAME is not the target name: two
  # add_test names can drive one binary (ninfer_gqa_split_geometry_probe_legacy/_exact differ
  # only by their ENVIRONMENT property), and a script test runs an interpreter with a script
  # that lives in the SOURCE tree. Keying on the name gave five false positives and hid the
  # real rows; the command is the only thing that says what a registration will execute.
  sed -n 's/^add_test(\[=\[\([^]]*\)\]=\] *"\([^"]*\)" *\(.*\))$/\1\t\2\t\3/p' \
    "$BUILD/tests/CTestTestfile.cmake" > /tmp/clh_exe.$$ 2>/dev/null
  n3a=0; n3b=0; nok=0
  : > /tmp/clh_unbuilt.$$
  while IFS=$'\t' read -r t e rest; do
    [ -n "$t" ] || continue
    b="$(basename "${e:-}")"
    case "$b" in
      bash|sh|dash|python|python3|perl)
        sc="$(printf '%s' "${rest:-}" | sed -n 's/^"\([^"]*\)".*/\1/p')"
        if [ -n "$sc" ] && [ -e "$sc" ]; then
          nok=$((nok+1))
        else
          printf 'NOTARGET\t%s\t%s\n' "$t" "script test but its script is missing: ${sc:-<none>}" \
            >> /tmp/clh_unbuilt.$$
          n3a=$((n3a+1))
        fi ;;
      "")
        printf 'NOTARGET\t%s\t%s\n' "$t" "no command at all" >> /tmp/clh_unbuilt.$$
        n3a=$((n3a+1)) ;;
      *)
        if [ -x "${e:-}" ]; then nok=$((nok+1)); continue; fi
        if [ -d "$BUILD/tests/CMakeFiles/$b.dir" ]; then
          d="$BUILD/tests/CMakeFiles/$b.dir"
          o=$(ls "$d"/*.o "$d"/*/*.o "$d"/*/*/*.o 2>/dev/null | grep -vc 'cmake_device_link\.o$')
          printf 'UNBUILT\t%s\t%s\t%s\n' "$t" "$o" "$b" >> /tmp/clh_unbuilt.$$
          n3b=$((n3b+1))
        else
          printf 'NOTARGET\t%s\t%s\n' "$t" "no executable, no CMake target" >> /tmp/clh_unbuilt.$$
          n3a=$((n3a+1))
        fi ;;
    esac
  done < /tmp/clh_exe.$$
  if [ "$n3a" -gt 0 ]; then
    echo "FAIL  3 registration: $n3a registered test name(s) can NEVER be built:"
    awk -F'\t' '$1=="NOTARGET"{print "         " $2 "   (" $3 ")"}' /tmp/clh_unbuilt.$$
    n3=$((n3 + n3a)); fail=1
  fi
  if [ "$n3b" -gt 0 ]; then
    if [ "$ALLOW_UNBUILT" = 1 ]; then
      echo "WARN  3 registration: $n3b registered test(s) have a target but NO binary (--allow-unbuilt):"
      warn=1
    else
      echo "FAIL  3 registration: $n3b registered test(s) have a CMake target but NO binary."
      echo "         A registration is not coverage: 'ctest -N' counts these among the total and"
      echo "         they can only ever report '***Not Run'. Build them, or pass --allow-unbuilt"
      echo "         to accept a mid-flight build directory out loud."
      n3=$((n3 + n3b)); fail=1
    fi
    awk -F'\t' '$1=="UNBUILT"{
        if ($3+0 > 0) printf "         %s   (%s object(s) present -> a link ran and FAILED)\n", $2, $3;
        else          printf "         %s   (no objects -> never built; building it is the only way to know if it links)\n", $2;
      }' /tmp/clh_unbuilt.$$
  fi
  if [ "$n3a" -eq 0 ] && [ "$n3b" -eq 0 ]; then
    echo "ok    3 registration: every registered test has a runnable artefact ($nok of them)"
  fi

echo "--"
echo "findings: archive-interface=$n1 missing-output=$n2 registration=$n3 | warnings: stale=$n4"
[ "$ninfo" -gt 0 ] && echo "info: $ninfo registered test target(s) not built yet (no objects, no output)"
rm -f /tmp/clh_reg.$$ /tmp/clh_disk.$$ /tmp/clh_exe.$$ /tmp/clh_unbuilt.$$ /tmp/clh_dirs.$$ /tmp/clh_stale.$$
if [ "$fail" -ne 0 ]; then echo "RESULT: FAIL"; exit 1; fi
if [ "$warn" -ne 0 ]; then echo "RESULT: PASS (with staleness warnings)"; exit 0; fi
echo "RESULT: PASS"
exit 0
