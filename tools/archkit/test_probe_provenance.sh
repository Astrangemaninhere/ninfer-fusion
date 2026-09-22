#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_probe_provenance.sh -- A SIMULATED ARM MAY NOT ENTER THE SUPPORT LIST THROUGH THE RAW.
#
# WHAT THIS IS FOR. tools/archkit/probe_formats.sh writes a format support list from a raw
# arm log. The simulator state is a property of a RUN, but a raw log is an append-only record
# of MANY runs: the driver recorded the state once, only when the raw file was CREATED, and
# resolved it at derive time with
#     if RAW_SIM non-empty -> use it ; elif banner -> fail closed
# so a stale `# probe_sim=none` made the banner test UNREACHABLE. MEASURED 2026-09-18: a raw
# created by a real run, then appended to by an NINFER_PROBE_ALLOW_SIM_ROUTE=1 run, derived
# meta source="real_gpu" with `supported=yes` rows harvested from the SIMULATED arm -- and
# src/core/format_probe.h ADMITTED that list as "MEASURED SUPPORTED".
#
# THE CONTRACT THIS PINS, per arm:
#   * a `SIMULATED ARCHITECTURE` banner anywhere in the raw  -> the whole list is simulated;
#   * any arm header recording `sim=` other than `none`      -> the whole list is simulated;
#   * every arm header recording `sim=none`                  -> the list is real (a POSITIVE
#     record, not an absence);
#   * a raw whose arm headers carry no `sim=` at all (one written before per-arm records
#     existed) is real ONLY if no banner is present in it, because the only writer of such a
#     raw invoked each arm as `out="$(NINFER_OP_REPORT_STATS=1 "$bin" 2>&1)"` (MEASURED:
#     BACKUP-SIMPROBE/probe_formats.sh.pre line 172) and so could not have discarded one;
#   * a simulated run may not APPEND to a raw of a non-simulated class (rc=11) -- that append
#     is what leaves the stale file-level line behind.
# A simulated list is written MARKED (source="simulated_route", every row
# leg=route_simulated supported=no simulated=yes sim_rung=<rung>) and the gate refuses it.
#
# NO GPU, NO LOCK. The derivation is a pure function of the raw log (the driver says so itself)
# and `--derive-only` acquires no lock: the lock acquisition is inside the run branch, which
# this test never enters except in T5 -- and T5's refusal is issued BEFORE that branch.
#
# usage:  test_probe_provenance.sh
# exit 0 iff every check passed.
# ---------------------------------------------------------------------------
set -u
REPO="${NINFER_REPO:-/home/user/ninfer-fusion}"
# NINFER_PROBE_DRIVER exists so this test can be run against a PRE-FIX copy of the driver to
# show that it FAILS: a test that cannot fail is not evidence. See the revert check in
# SIMFIX/REPORT.md.
DRIVER="${NINFER_PROBE_DRIVER:-$REPO/tools/archkit/probe_formats.sh}"
WORK="${SIMFIX_TEST_DIR:-/home/user/scratch/PATCHSET/SIMFIX/provtest}"
# THIS TEST DELETES AND RECREATES $WORK ON EVERY RUN, and since it is registered in `ctest` that
# now happens in normal use rather than only when a human types it. So an overridable path here is
# a footgun: an exported SIMFIX_TEST_DIR=/ (or /home) would have a TEST delete it. Refuse anything
# that is not an absolute path at least three components deep, and say why on stderr so that
# `ctest --output-on-failure` shows the reason.
case "$WORK" in
  /*) ;;
  *) echo "test_probe_provenance: SIMFIX_TEST_DIR must be an ABSOLUTE path, got '$WORK'" >&2
     exit 2 ;;
esac
WORK_DEPTH="$(printf '%s' "$WORK" | tr -cd '/' | wc -c)"
if [ "$WORK_DEPTH" -lt 3 ]; then
  echo "test_probe_provenance: refusing to rm -rf '$WORK' (only $WORK_DEPTH path component(s))." >&2
  echo "  This test deletes and recreates its work dir; a shallow path is a footgun." >&2
  exit 2
fi
KEY='v1;ki=1;bf16=1;fp16=1;i8=1;f8f6f4=1;mxf4=1;smr=1'
DECL='name=NVIDIA GeForce RTX 5090 D;cc=12.0;build=120a'
BANNER='ninfer: *** SIMULATED ARCHITECTURE -- TEST ONLY ***  physical sm_120, simulated sm_70; the kernels that will execute are the ones THIS binary was compiled for, NOT sm_70 binaries'

CHECKS=0
FAILS=0
ok()   { CHECKS=$((CHECKS+1)); printf '  ok   %s\n' "$1"; }
bad()  { CHECKS=$((CHECKS+1)); FAILS=$((FAILS+1)); printf '  FAIL %s\n' "$1"; }
# check <0|1> <label> [detail] -- the optional third argument is printed ON FAILURE ONLY, so a
# failing check can say what to do about it without a separate branch at every call site.
check(){ if [ "$1" = "0" ]; then ok "$2"; else bad "$2${3:+  [$3]}"; fi; }

rm -rf "$WORK"; mkdir -p "$WORK"
cd "$REPO" || exit 1

# --- raw builders: the driver's own format, built from its own emitted field shapes ----------
raw_header() {  # $1 = path, $2 = the file-level `# probe_sim=` value ("" writes none)
  {
    printf '# probe_formats.sh raw arm log -- appended, never rewritten\n'
    printf '# probe_key=%s\n' "$KEY"
    printf '# probe_declared=%s  (DECLARED, informational)\n' "$DECL"
    printf '# git HEAD 0000000000000000000000000000000000000000\n'
    printf '# arch_suffix=120a lib_sha16=deadbeefdeadbeef host=NVIDIA GeForce RTX 5090 D, 12.0\n'
    [ -n "$2" ] && printf '# probe_sim=%s\n' "$2"
  } > "$1"
}
# an arm whose band 17..open passes the criterion (so supported=yes is DERIVABLE from it)
arm_real() {  # $1 = raw, $2 = fmt, $3 = path, $4 = extra " sim=..." or ""
  {
    printf '##### ARM %s path=%s target=ninfer_linear_x_test rc=0 sha16=e3cbdc61e8f64296 elapsed=13s%s\n' "$2" "$3" "$4"
    printf 'OP_ERROR_STATS kind=reduction count=14336 rel_l2=0.0016 rel_l2_limit=0.00390625 rel_l2_ratio=0.42 rmse=0.0012 non_finite=0 case=%s Linear [14336,5120] T=17\n' "$2"
    printf 'OP_ERROR_STATS kind=reduction count=14336 rel_l2=0.0019 rel_l2_limit=0.00390625 rel_l2_ratio=0.49 rmse=0.0014 non_finite=0 case=%s Linear [14336,5120] T=1024\n' "$2"
  } >> "$1"
}
# an arm that ran under the simulator, optionally carrying the banner in its captured output
arm_sim() {   # $1 = raw, $2 = fmt, $3 = path, $4 = banner|nobanner
  {
    printf '##### ARM %s path=%s target=ninfer_linear_y_test rc=0 sha16=798adae67805d003 elapsed=5s sim=active:70\n' "$2" "$3"
    [ "$4" = "banner" ] && printf '%s\n' "$BANNER"
    printf 'OP_ERROR_STATS kind=reduction count=14336 rel_l2=0.0017 rel_l2_limit=0.00390625 rel_l2_ratio=0.44 rmse=0.0013 non_finite=0 case=%s Linear [14336,5120] T=17\n' "$2"
  } >> "$1"
}

derive() {  # $1 = raw path; the list goes to ${1%.raw}
  local raw="$1" out="${1%.raw}"
  rm -f "$out"
  env -u NINFER_SIM_ARCH -u NINFER_SIM_ARCH_ACK -u NINFER_PROBE_ALLOW_SIM_ROUTE \
      NINFER_PROBE_KEY="$KEY" NINFER_PROBE_DECLARED="$DECL" \
      "$DRIVER" "$KEY" "$out" --derive-only > "$raw.drvout" 2> "$raw.drverr"
  printf '%s' "$?"
}
val() { sed -n "s/^meta $2=\"\(.*\)\"$/\1/p" "$1" | head -1; }
n_sup()   { grep -c '^entry .*supported=yes' "$1" || true; }
n_sim()   { grep -c '^entry .*simulated=yes' "$1" || true; }
n_route() { grep -c '^entry .*leg=route_simulated' "$1" || true; }
n_ent()   { grep -c '^entry ' "$1" || true; }
allrows() { [ "$(n_sim "$1")" = "$(n_ent "$1")" ]; }

echo "== test_probe_provenance: the raw log's provenance is per ARM, not per file =="
echo "driver sha16 = $(sha256sum "$DRIVER" | cut -c1-16)"

# ---------------------------------------------------------------------------------------
# T0  THE DRIVER MUST BE EXECUTABLE, AND THIS IS THE FIRST CHECK FOR A REASON.
#
# An edit script that writes a temp file and os.replace()s it does NOT carry the file mode, so
# 0755 silently becomes 0644. Nothing else notices: `bash -n` still passes, the file still reads
# correctly, and the failure shows up as rc=126 on every invocation -- a code that looks like a
# driver bug and is not one. MEASURED 2026-09-18 (SIMFIX edit 1): that is how this line lost a
# debugging round. A check rather than a comment, because a comment is read and a check is run.
# ---------------------------------------------------------------------------------------
echo "T0 the driver under test is a runnable file"
check "$([ -f "$DRIVER" ] && echo 0 || echo 1)" \
  "T0 driver exists at $DRIVER" "point NINFER_PROBE_DRIVER at the driver you mean to test"
check "$([ -x "$DRIVER" ] && echo 0 || echo 1)" \
  "T0 the driver is EXECUTABLE -- an os.replace-based edit drops 0755 and every call returns 126" \
  "mode=$(stat -c%a "$DRIVER" 2>/dev/null) -- chmod 755 it"

# ===========================================================================
# T1  THE FINDING. A real raw (file-level record `none`), then a simulated arm APPENDED,
#     whose captured output carries the banner. Before the fix this derived
#     source="real_gpu" with supported=yes rows and the gate ADMITTED it.
# ===========================================================================
R="$WORK/t1.list.raw"
raw_header "$R" "none"; arm_real "$R" BF16 A16 " sim=none"; arm_sim "$R" NVFP4 A4 banner
rc="$(derive "$R")"; L="$WORK/t1.list"
echo "T1 stale file-level record + a simulated arm whose output carries the banner"
check "$([ "$rc" = 0 ] && echo 0 || echo 1)" "T1 driver rc=0 (got $rc)"
check "$([ "$(val "$L" source)" = simulated_route ] && echo 0 || echo 1)" "T1 meta source=simulated_route (got '$(val "$L" source)')"
check "$([ "$(n_sup "$L")" = 0 ] && echo 0 || echo 1)" "T1 no row claims supported=yes (got $(n_sup "$L"))"
check "$([ "$(n_route "$L")" = "$(n_ent "$L")" ] && echo 0 || echo 1)" "T1 every row leg=route_simulated ($(n_route "$L")/$(n_ent "$L"))"
check "$(allrows "$L" && echo 0 || echo 1)" "T1 every row simulated=yes ($(n_sim "$L")/$(n_ent "$L"))"
check "$([ "$(val "$L" sim_run_rung)" = 70 ] && echo 0 || echo 1)" "T1 sim_rung=70 (got '$(val "$L" sim_run_rung)')"
check "$([ -n "$(val "$L" sim_arms)" ] && echo 0 || echo 1)" "T1 meta sim_arms records every arm ('$(val "$L" sim_arms)')"
check "$([ "$(val "$L" provenance_resolved_by)" = banner-in-raw ] && echo 0 || echo 1)" "T1 provenance resolved by the banner (got '$(val "$L" provenance_resolved_by)')"

# ===========================================================================
# T2  THE RESIDUAL THE BANNER ALONE CANNOT CATCH. Same stale file-level record, and the
#     simulated arm's output does NOT carry the banner. Only the per-arm record can decide.
# ===========================================================================
R="$WORK/t2.list.raw"
raw_header "$R" "none"; arm_real "$R" BF16 A16 " sim=none"; arm_sim "$R" NVFP4 A4 nobanner
rc="$(derive "$R")"; L="$WORK/t2.list"
echo "T2 stale file-level record + a simulated arm with NO banner in its output"
check "$([ "$rc" = 0 ] && echo 0 || echo 1)" "T2 driver rc=0 (got $rc)"
check "$([ "$(val "$L" source)" = simulated_route ] && echo 0 || echo 1)" "T2 meta source=simulated_route (got '$(val "$L" source)')"
check "$([ "$(n_sup "$L")" = 0 ] && echo 0 || echo 1)" "T2 no row claims supported=yes (got $(n_sup "$L"))"
check "$(allrows "$L" && echo 0 || echo 1)" "T2 every row simulated=yes ($(n_sim "$L")/$(n_ent "$L"))"
check "$([ "$(val "$L" provenance_resolved_by)" = per-arm-record ] && echo 0 || echo 1)" "T2 provenance resolved by the per-arm record (got '$(val "$L" provenance_resolved_by)')"

# ===========================================================================
# T3  THE CONTROL, so T1/T2 are not vacuous. Every arm records sim=none, no banner:
#     the list MUST derive real_gpu and MUST be able to carry supported=yes.
# ===========================================================================
R="$WORK/t3.list.raw"
raw_header "$R" "none"; arm_real "$R" BF16 A16 " sim=none"; arm_real "$R" NVFP4 A4 " sim=none"
rc="$(derive "$R")"; L="$WORK/t3.list"
echo "T3 control: a raw whose arms all record sim=none"
check "$([ "$rc" = 0 ] && echo 0 || echo 1)" "T3 driver rc=0 (got $rc)"
check "$([ "$(val "$L" source)" = real_gpu ] && echo 0 || echo 1)" "T3 meta source=real_gpu (got '$(val "$L" source)')"
check "$([ "$(n_sim "$L")" = 0 ] && echo 0 || echo 1)" "T3 no row is marked simulated (got $(n_sim "$L"))"
check "$([ "$(n_sup "$L")" -ge 1 ] && echo 0 || echo 1)" "T3 a real raw can still derive supported=yes (got $(n_sup "$L"))"
check "$([ "$(val "$L" provenance_resolved_by)" = per-arm-record-all-none ] && echo 0 || echo 1)" "T3 provenance resolved by the all-none record (got '$(val "$L" provenance_resolved_by)')"

# ===========================================================================
# T4  the box's REAL raw and its REAL list must still work (the predecessor's control).
# ===========================================================================
R="$REPO/tools/archkit/probe_results/format_support-120.list.raw"
L="$WORK/t4.list"; cp -f "$R" "$L.raw"
rc="$(derive "$L.raw")"
echo "T4 the box's own raw log (no per-arm record, no banner: a legacy raw)"
check "$([ "$rc" = 0 ] && echo 0 || echo 1)" "T4 driver rc=0 (got $rc)"
check "$([ "$(val "$L" source)" = real_gpu ] && echo 0 || echo 1)" "T4 meta source=real_gpu (got '$(val "$L" source)')"
check "$([ "$(n_sim "$L")" = 0 ] && echo 0 || echo 1)" "T4 no row is marked simulated (got $(n_sim "$L"))"
check "$([ "$(n_sup "$L")" -ge 1 ] && echo 0 || echo 1)" "T4 the real raw still derives supported=yes (got $(n_sup "$L"))"
check "$([ "$(val "$L" provenance_resolved_by)" = legacy-no-record-no-banner ] && echo 0 || echo 1)" "T4 provenance resolved as a legacy raw (got '$(val "$L" provenance_resolved_by)')"

# ===========================================================================
# T7  MARK AND REFUSE, NOT REFUSE-OUTRIGHT, FOR A READ-ONLY PASS. `--derive-only` appends no
#     arm, so it can mix nothing; running it with the simulator set in the environment must
#     still answer with a MARKED list (the measured rows kept), not with the append refusal.
# ===========================================================================
R="$WORK/t7.list.raw"
raw_header "$R" "none"; arm_real "$R" BF16 A16 " sim=none"; arm_real "$R" NVFP4 A4 " sim=none"
L="$WORK/t7.list"; rm -f "$L"
env NINFER_SIM_ARCH=70 NINFER_SIM_ARCH_ACK=I-UNDERSTAND-THIS-IS-NOT-A-V100 \
    NINFER_PROBE_KEY="$KEY" NINFER_PROBE_DECLARED="$DECL" \
    "$DRIVER" "$KEY" "$L" --derive-only > "$WORK/t7.out" 2> "$WORK/t7.err"
rc=$?
echo "T7 --derive-only over a REAL raw with the simulator in the environment"
check "$([ "$rc" = 0 ] && echo 0 || echo 1)" "T7 driver rc=0, not the append refusal (got $rc)"
check "$([ "$(val "$L" source)" = simulated_route ] && echo 0 || echo 1)" "T7 the list is written MARKED (got '$(val "$L" source)')"
check "$([ "$(n_sup "$L")" = 0 ] && echo 0 || echo 1)" "T7 no row claims supported=yes (got $(n_sup "$L"))"
check "$(allrows "$L" && echo 0 || echo 1)" "T7 every row is marked simulated=yes ($(n_sim "$L")/$(n_ent "$L"))"
check "$([ "$(val "$L" sim_run_rung)" = 70 ] && echo 0 || echo 1)" "T7 the rung is named (got '$(val "$L" sim_run_rung)')"
check "$(grep -q 'REFUSING TO APPEND' "$WORK/t7.err" && echo 1 || echo 0)" "T7 the append refusal did NOT fire"

# ===========================================================================
# T5  a simulated run may not APPEND to a raw of another class: that append is what leaves
#     the stale file-level record behind. Must refuse rc=11, write nothing, and take NO lock.
#
#     GATED BY AN EXPLICIT ENV SWITCH, and the gate is named rather than silent. This is the
#     only check that enters the driver's RUN branch, and against a driver that predates the
#     rc=11 refusal that branch would proceed to acquire the project's GPU lock and run arms
#     -- a shared resource taken to test a driver that is not in the tree. So the revert check
#     sets SIMFIX_SKIP_T5=1 and reports this check as SKIPPED, never as passed.
# ===========================================================================
if [ "${SIMFIX_SKIP_T5:-0}" = "1" ]; then
  echo "T5 SKIPPED (SIMFIX_SKIP_T5=1): the driver under test is not the tree's, and T5 is the"
  echo "   only check that would enter the run branch and take the GPU lock."
else
R="$WORK/t5.list.raw"
raw_header "$R" "none"; arm_real "$R" BF16 A16 " sim=none"
OUT="$WORK/t5.list"; rm -f "$OUT"
env NINFER_SIM_ARCH=70 NINFER_SIM_ARCH_ACK=I-UNDERSTAND-THIS-IS-NOT-A-V100 NINFER_PROBE_ALLOW_SIM_ROUTE=1 \
    NINFER_PROBE_KEY="$KEY" NINFER_PROBE_DECLARED="$DECL" \
    "$DRIVER" "$KEY" "$OUT" > "$WORK/t5.out" 2> "$WORK/t5.err"
rc=$?
echo "T5 an ALLOW_SIM_ROUTE run appending to a real raw"
check "$([ "$rc" = 11 ] && echo 0 || echo 1)" "T5 driver rc=11 (got $rc)"
check "$([ ! -e "$OUT" ] && echo 0 || echo 1)" "T5 nothing was written to the list"
check "$(grep -q 'REFUSING TO APPEND' "$WORK/t5.err" && echo 0 || echo 1)" "T5 the refusal says REFUSING TO APPEND"
check "$(grep -q 'fresh raw per provenance class' "$WORK/t5.err" && echo 0 || echo 1)" "T5 the refusal carries the on-ramp"
check "$(grep -q 'NO GPU LOCK' "$WORK/t5.err" && echo 1 || echo 0)" "T5 refused BEFORE the GPU lock was attempted"
check "$([ "$(grep -c '^##### ARM ' "$R")" = 1 ] && echo 0 || echo 1)" "T5 the raw was not appended to"

# T8  the same refusal for the LEGACY class -- the box's own raw has no per-arm record, and
#     appending a simulated arm to it is exactly the append that produced the stale line.
RL="$WORK/t8.list.raw"
cp -f "$REPO/tools/archkit/probe_results/format_support-120.list.raw" "$RL"
OUT="$WORK/t8.list"; rm -f "$OUT"
arms_before="$(grep -c '^##### ARM ' "$RL")"
env NINFER_SIM_ARCH=70 NINFER_SIM_ARCH_ACK=I-UNDERSTAND-THIS-IS-NOT-A-V100 NINFER_PROBE_ALLOW_SIM_ROUTE=1 \
    NINFER_PROBE_KEY="$KEY" NINFER_PROBE_DECLARED="$DECL" \
    "$DRIVER" "$KEY" "$OUT" > "$WORK/t8.out" 2> "$WORK/t8.err"
rc=$?
echo "T8 an ALLOW_SIM_ROUTE run appending to a LEGACY raw (a copy of the box's own)"
check "$([ "$rc" = 11 ] && echo 0 || echo 1)" "T8 driver rc=11 (got $rc)"
check "$([ ! -e "$OUT" ] && echo 0 || echo 1)" "T8 nothing was written to the list"
check "$(grep -q "class 'legacy'" "$WORK/t8.err" && echo 0 || echo 1)" "T8 the refusal names the raw's class"
check "$([ "$(grep -c '^##### ARM ' "$RL")" = "$arms_before" ] && echo 0 || echo 1)" "T8 the legacy raw was not appended to"
check "$(grep -q 'NO GPU LOCK' "$WORK/t8.err" && echo 1 || echo 0)" "T8 refused BEFORE the GPU lock was attempted"
fi

# ===========================================================================
# T6  THE GATE. The lists T1/T2 produced must be REFUSED by src/core/format_probe.h, not
#     merely marked. This is the seam the defect crossed, so it is checked, not assumed.
# ===========================================================================
BIN="$WORK/test_sim_no_support_bin"
if [ ! -x "$BIN" ]; then
  g++ -std=c++20 -O1 -Wall -Wextra -I "$REPO/src" -I "$REPO/include" -I "$REPO/tests" -I "$REPO/third_party" \
      -DNINFER_HAVE_QPN=1 -DNINFER_SOURCE_DIR="\"$REPO\"" \
      "$REPO/tests/test_sim_no_support.cpp" \
      "$REPO/build/src/libninfer_artifact.a" "$REPO/build/src/libninfer_core.a" \
      -o "$BIN" 2> "$WORK/t6_compile.err"
fi
if [ ! -x "$BIN" ]; then
  echo "T6 CANNOT RUN: the gate binary did not build. This is NOT a pass:"
  sed -n '1,10p' "$WORK/t6_compile.err"
  FAILS=$((FAILS+1)); CHECKS=$((CHECKS+1))
else
  for t in t1 t2; do
    "$BIN" "$WORK/$t.list" > "$WORK/$t.gate" 2>&1
    grc=$?
    printf '  ... %s gate rc=%d: %s\n' "$t" "$grc" "$(grep 'VERDICT' "$WORK/$t.gate" | head -1)"
    check "$([ "$grc" = 0 ] && echo 0 || echo 1)" "T6 $t: the gate REFUSES the list as support (rc=$grc, 0=refused)"
  done
  # the other direction, so T6 is not vacuous: the real list must still be ADMITTED
  "$BIN" "$REPO/tools/archkit/probe_results/format_support-120.list" > "$WORK/t6c.gate" 2>&1
  crc=$?
  check "$([ "$crc" = 1 ] && echo 0 || echo 1)" "T6 control: the box's real list is still ADMITTED (rc=$crc, 1=admitted)"
fi

echo
printf 'test_probe_provenance: %d checks, %d failure(s)\n' "$CHECKS" "$FAILS"
[ "$FAILS" = 0 ] || exit 1
echo "OK  test_probe_provenance: a simulated arm cannot reach the support list through the raw"
