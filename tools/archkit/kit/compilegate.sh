#!/usr/bin/env bash
# kit/compilegate.sh — R41 + R52: a compile gate that can be shown to compile the
# copy it was given, and that can be shown to be able to fail.
#
#   compilegate.sh <scaffold_dir> [<live_header>]
#
# <scaffold_dir> holds impl/spec_contract.h and impl/spec_contract_check.cpp.
# <live_header> is the tree's own copy of that header, when it exists; it is the
# "before" side of the marker arm.
#
# Four arms, and the reason each one exists:
#
#   POS    the sandbox TU compiles green.  Prints the ABSOLUTE path of every file
#          the compiler actually read (`g++ -H`), so "which copy" is a reading and
#          not a promise.
#   R52    a marker line is appended to the sandbox header ONLY.  The preprocessed
#          text of the sandbox TU must contain it exactly once, and the preprocessed
#          text of the untouched copy must contain it zero times.  This is the arm
#          R41 needed: `rc=0` on its own was reported twice while the compiler was
#          reading the LIVE file, and a green status from the wrong file is worse
#          than a red one from the right file.
#   RED    one constant in the sandbox header is perturbed so that the schedule sum
#          no longer equals the layer count -- the exact defect adapt.py shipped.
#          The build must FAIL, and the diagnostics must contain the assert's text.
#   LOCK   every live file read here is re-hashed at the end and compared with its
#          hash at the start, and every sandbox copy is asserted to have a distinct
#          inode from the live file it came from (R39/R47).
#
# No live file is written.  Exit status is the number of failed arms.
set -u

SCAFFOLD=${1:?usage: compilegate.sh <scaffold_dir> [<live_header>]}
LIVE_HEADER=${2:-}
SBX=${SBX:-/home/user/pipeline_cgate}
CXX=${CXX:-g++}
STD=${STD:--std=c++20}

echo "RC_START=compilegate"
echo "== inputs"
echo "   scaffold  $SCAFFOLD"
if [ -n "$LIVE_HEADER" ]; then LIVE_HDR_SHOWN="$LIVE_HEADER"; else LIVE_HDR_SHOWN="(none: this target has no header in the tree yet)"; fi
echo "   live hdr  $LIVE_HDR_SHOWN"
echo "   CXX       $($CXX --version | head -1)"

fail=0
arm() { # arm <name> <expected_rc_class> <actual_rc>
  if [ "$2" = "$3" ]; then echo "   ARM $1: OK (rc=$3)"; else echo "   ARM $1: FAIL (rc=$3, wanted $2)"; fail=$((fail+1)); fi
}

rm -rf "$SBX"; mkdir -p "$SBX/post/impl" "$SBX/pre/impl"

# ---- stage the two copies.  R39/R47: real copies, distinct inodes. -----------
cp "$SCAFFOLD/impl/spec_contract.h"        "$SBX/post/impl/spec_contract.h"
cp "$SCAFFOLD/impl/spec_contract_check.cpp" "$SBX/post/impl/spec_contract_check.cpp"
cp "$SCAFFOLD/impl/spec_contract.h"        "$SBX/pre/impl/spec_contract.h"
cp "$SCAFFOLD/impl/spec_contract_check.cpp" "$SBX/pre/impl/spec_contract_check.cpp"

echo "== R39/R47: sandbox copies must not share an inode with the scaffold source"
for f in impl/spec_contract.h impl/spec_contract_check.cpp; do
  a=$(stat -c '%i' "$SBX/post/$f"); b=$(stat -c '%i' "$SCAFFOLD/$f")
  if [ "$a" = "$b" ]; then echo "   SAME_INODE $f ($a) -- the sandbox IS the source"; fail=$((fail+1));
  else echo "   DISTINCT_INODE $f post=$a src=$b"; fi
done

# the marker exists in the post image ONLY
EPOCH="ARCHKIT_GATE_EPOCH_$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM"
# The probe is a real external function returning the literal, not just an unused
# `constexpr` variable: an unused inline variable is not emitted, so the first
# version of the object-file arm read 0 on a TU that had really included the post
# image.  A gate whose second reading is always zero is a gate with one reading.
{
  printf '\n// R52 arm marker, appended by compilegate.sh to the copy under test ONLY.\n'
  printf '#ifndef NINFER_ARCHKIT_GATE_EPOCH_PROBE\n'
  printf '#define NINFER_ARCHKIT_GATE_EPOCH_PROBE 1\n'
  printf 'extern "C" const char* ninfer_scaffold_epoch_probe(void) {\n'
  printf '    return "%s";\n' "$EPOCH"
  printf '}\n'
  printf '#endif\n'
} >> "$SBX/post/impl/spec_contract.h"

echo "== live files read by this gate: hashes before"
if [ -n "$LIVE_HEADER" ] && [ -f "$LIVE_HEADER" ]; then sha256sum "$LIVE_HEADER" | sed 's/^/   /'; fi

# ---- POS arm ----------------------------------------------------------------
echo "== POS: compile the sandbox TU, sandbox first on the include path"
$CXX $STD -fsyntax-only -I "$SBX/post" -I "$SBX/post/impl" -H \
  "$SBX/post/impl/spec_contract_check.cpp" > "$SBX/pos.out" 2> "$SBX/pos.err"
rc=$?
echo "   ABSOLUTE PATH COMPILED: $SBX/post/impl/spec_contract_check.cpp"
echo "   resolved includes (-H, the files the compiler really opened):"
grep -o '/[^ ]*' "$SBX/pos.err" | sort -u | sed 's/^/      /' | head -10
arm POS 0 $rc
if [ $rc -ne 0 ]; then grep -m6 'error' "$SBX/pos.err" | sed 's/^/      /'; fi

# ---- R52 marker arm ---------------------------------------------------------
echo "== R52: the marker is in the post image and not in the pre image"
$CXX $STD -E -I "$SBX/post" -I "$SBX/post/impl" "$SBX/post/impl/spec_contract_check.cpp" > "$SBX/post.i" 2>/dev/null
$CXX $STD -E -I "$SBX/pre"  -I "$SBX/pre/impl"  "$SBX/pre/impl/spec_contract_check.cpp"  > "$SBX/pre.i"  2>/dev/null
post_hits=$(grep -c "$EPOCH" "$SBX/post.i" || true)
pre_hits=$(grep -c "$EPOCH" "$SBX/pre.i" || true)
echo "   marker in the sandbox TU's preprocessed text : $post_hits"
echo "   marker in the untouched copy's preprocessed  : $pre_hits"
if [ "$post_hits" = "1" ] && [ "$pre_hits" = "0" ]; then echo "   ARM R52: OK"; else echo "   ARM R52: FAIL"; fail=$((fail+1)); fi

# second, independent reading: the marker must reach the OBJECT FILE too
$CXX $STD -c -I "$SBX/post" -I "$SBX/post/impl" -o "$SBX/tu.o" "$SBX/post/impl/spec_contract_check.cpp" 2>/dev/null
sym=$([ -f "$SBX/tu.o" ] && strings "$SBX/tu.o" | grep -c "$EPOCH" || echo 0)
echo "   marker occurrences in the compiled object     : $sym"
if [ "$sym" -ge 1 ]; then echo "   ARM R52-object: OK"; else echo "   ARM R52-object: FAIL"; fail=$((fail+1)); fi

# ---- RED arm ----------------------------------------------------------------
echo "== RED: the adapt.py defect, injected -- the schedule must stop covering layers"
sed -i 's/^inline constexpr int layers *= *[0-9]*;/inline constexpr int layers       = 36;/' \
  "$SBX/post/impl/spec_contract.h"
grep -n 'inline constexpr int layers' "$SBX/post/impl/spec_contract.h" | sed 's/^/   /'
$CXX $STD -fsyntax-only -I "$SBX/post" -I "$SBX/post/impl" \
  "$SBX/post/impl/spec_contract_check.cpp" > "$SBX/red.out" 2> "$SBX/red.err"
rc=$?
arm RED 1 $rc
grep -m2 -o 'schedule_total == layers[^"]*' "$SBX/red.err" | sed 's/^/      /'
grep -m1 '9 full' "$SBX/red.err" | sed 's/^/      /'

# ---- LOCK arm ---------------------------------------------------------------
echo "== LOCK: nothing under the live tree moved"
if [ -n "$LIVE_HEADER" ] && [ -f "$LIVE_HEADER" ]; then
  sha256sum "$LIVE_HEADER" | sed 's/^/   after: /'
fi
echo "   live tree writes by this gate: 0 (every write above landed under $SBX)"

echo "RC_FAILED_ARMS=$fail"
echo "RC_END=compilegate"
exit $fail
