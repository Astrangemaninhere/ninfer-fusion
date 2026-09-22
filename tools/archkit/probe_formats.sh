#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# probe_formats.sh -- THE PROBE SERIES, and the only writer of a format support list.
#
# usage:  probe_formats.sh <arch> <out.list> [--derive-only]
#
# WHAT IT MEASURES. For each (format, activation path, M band) it runs the repo's OWN
# per-format linear Op test with NINFER_OP_REPORT_STATS=1 and harvests the per-case
# `OP_ERROR_STATS kind=reduction ... case=...` lines that suite emits on SUCCESS as well as
# on failure. Those lines carry the measured relative-L2 and the criterion's limit, so the
# verdict recorded here is a number compared against a stated criterion -- NOT an exit code,
# and not the absence of an error.
#
# WHY NOT A LAUNCH-OR-NOT PROBE. "did it launch without error" admits a silently wrong
# kernel, and that is this project's most expensive failure mode (see the opening comment of
# src/core/format_probe.h for the measured incidents this rule exists because of).
#
# THE KEY IS THE MEASURED CAPABILITY VECTOR, NOT A NAME AND NOT A cc NUMBER. A declared
# attribute may be false -- a modified VBIOS can make a card present a listed name over
# different silicon -- so NINFER_PROBE_KEY must be supplied by whoever ran the engine's own
# known-answer probes (src/core/device_probe.cu, reported as the key that
# src/core/format_probe.h's measured_capability_key() renders), and NINFER_PROBE_DECLARED
# records what the card SAID. This script refuses to run without a key rather than inventing
# one from the build arch: an invented key would be a declaration wearing a measurement's name.
#
# THE CRITERION IS QUOTED, NOT INVENTED: tests/ops/linear/linear_test_common.cpp:38-48
# (tolerance_for) against cpu_linear_gemm_fp64 (:246). The activation path (A16/A8/A4)
# selects it; it is recorded in the emitted list.
#
# WHAT IT REFUSES TO DO:
#   * it will not write supported=yes for a band with zero measured cases (M 9..16 on the
#     NVFP4 W4A4 arm is the live example: the suite's cases are T=1,2,4,5,8,17,1024);
#   * it will not write supported=yes for a band whose ARM was not green, even when every
#     measured case in that band passed: an arm that faults elsewhere is a defect that must
#     not be laundered into a support verdict (measured 2026-09-18: ninfer_linear_q4_a16_test
#     hits cudaErrorIllegalAddress at [24576,4096] T=1 while its other cases pass);
#   * it will not run anything without the project's GPU lock;
#   * it will not run under the test-only architecture simulator without saying so: by default it
#     REFUSES (rc=10) and names the command that would produce a real row, because a simulated
#     ROUTE is not this card's routing; with NINFER_PROBE_ALLOW_SIM_ROUTE=1 it writes
#     meta source="simulated_route" and marks every row leg=route_simulated simulated=yes
#     supported=no sim_rung=<rung>, a shape src/core/format_probe.h refuses as support. (This
#     bullet previously claimed a staleness FLAG that echoed into `source`; no such flag exists in
#     this script, and the staleness decision is the gate's, not the writer's.)
#   * it will not APPEND an arm to a raw of a different provenance class (rc=11), because the
#     simulator state is a property of a RUN while a raw log is an append-only record of MANY
#     runs: mixing them leaves a file-level `# probe_sim=` that describes a run other than the
#     one behind some of the arms, and a derivation that trusted it wrote a real_gpu list out of
#     simulated rows (MEASURED 2026-09-18 -- such a list was ADMITTED as support). Every arm
#     header now carries `sim=<state>`, and the derivation reads the arms and the banner, never
#     one file-level line.
#   * a FAIL in any case of a band marks the band unsupported and the script still writes the
#     list, because a negative result is a result.
#
# IT IS RESUMABLE and the derivation is separable: an arm already present in <out>.raw is
# skipped, and `--derive-only` re-derives the list from an existing raw log without taking
# the GPU at all (the derivation is a pure function of the log).
set -u
export PATH="/home/user/.local/bin:/home/user/bin:$PATH"

ARCH="${1:?usage: probe_formats.sh <measured-capability-key> <out.list> [--derive-only]}"
OUT="${2:?usage: probe_formats.sh <key> <out.list> [--derive-only]}"
shift 2 || true
DERIVE_ONLY=0
for arg in "$@"; do
  [ "$arg" = "--derive-only" ] && DERIVE_ONLY=1
done

REPO="${NINFER_REPO:-/home/user/ninfer-fusion}"
LOG="${NINFER_PROBE_LOG:-/home/user/scratch/PATCHSET/PROBE2ROUTE/probe_run.log}"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
: >>"$LOG"

# ---------------------------------------------------------------------------
# The arm table. One row per (format, activation path, test target). This is the ONLY
# hand-written part of the file: which binary exercises which format. Everything a verdict
# depends on -- the numbers, the bands, the provenance -- is derived below.
# ---------------------------------------------------------------------------
ARMS=(
  "BF16|A16|ninfer_linear_bf16_a16_test"
  "Q4G64_F16S|A16|ninfer_linear_q4_a16_test"
  "Q5G64_F16S|A16|ninfer_linear_q5_a16_test"
  "Q6G64_F16S|A16|ninfer_linear_q6_a16_test"
  "W8G32_F16S|A16|ninfer_linear_w8_a16_test"
  "NVFP4|A16|ninfer_linear_nvfp4_a16_test"
  "NVFP4|A4|ninfer_linear_nvfp4_a4_test"
  "FP8_E4M3FN_ROW_BF16S|A16|ninfer_linear_fp8_a16_test"
  "FP8_E4M3FN_ROW_BF16S|A8|ninfer_linear_fp8_a8_test"
)

say() { printf '[probe %s] %s\n' "$(date '+%T')" "$*" | tee -a "$LOG" >&2; }

RAW="$OUT.raw"
BIN_DIR="$REPO/build/tests"
LIB="$REPO/build/src/libninfer_ops.a"
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
ARCH_SUFFIX="$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=//p' "$REPO/build/CMakeCache.txt" 2>/dev/null | head -1)"
ARCH_NUM="${ARCH_SUFFIX%%a}"
ARCH_SUFFIX_ONLY=""
[ -n "$ARCH_SUFFIX" ] && [ "$ARCH_SUFFIX" != "$ARCH_NUM" ] && ARCH_SUFFIX_ONLY="${ARCH_SUFFIX#$ARCH_NUM}"
HOST_GPU="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
# THE MEASURED KEY. Supplied, never derived: see the header. The probe arms below are what
# MEASURE the formats; the capability vector has to come from the engine's own known-answer
# capability probe, and if it is absent this script STRINGUALLY DECLARES nothing.
PROBE_KEY="${NINFER_PROBE_KEY:-}"
DECLARED="${NINFER_PROBE_DECLARED:-}"
# A --derive-only pass must carry the SAME key the arms were measured under, or the
# list would be re-keyed by a later read; the raw log records it, so it is read back
# from there. Done in ONE place, after RAW is known.
if [ -f "$RAW" ]; then
  [ -z "$PROBE_KEY" ] && PROBE_KEY="$(sed -n 's/^# probe_key=//p' "$RAW" | head -1)"
  [ -z "$DECLARED" ] && DECLARED="$(sed -n 's/^# probe_declared=//p' "$RAW" | head -1)"
fi
DECLARED_NAME="${DECLARED%%;*}"
DECLARED_NAME="${DECLARED_NAME#name=}"
DECLARED_CC="$(printf '%s' "$DECLARED" | tr ';' '\n' | sed -n 's/^cc=//p')"
DECLARED_BUILD="$(printf '%s' "$DECLARED" | tr ';' '\n' | sed -n 's/^build=//p')"
LIB_SHA="$(sha256sum "$LIB" 2>/dev/null | cut -c1-16)"
LIB_MTIME="$(date -r "$LIB" '+%F %T' 2>/dev/null)"

if [ "$DERIVE_ONLY" = "0" ] && [ -z "$PROBE_KEY" ]; then
  say "REFUSING TO RUN: NINFER_PROBE_KEY is not set."
  say "This probe keys its rows on the MEASURED capability vector, and the only instrument"
  say "that measures it is the engine's own known-answer capability probe"
  say "(src/core/device_probe.cu -> src/core/device_capabilities.h). Get the key from that"
  say "report -- src/core/format_probe.h's measured_capability_key() renders it, e.g."
  say "  v1;ki=1;bf16=1;fp16=1;i8=1;f8f6f4=1;mxf4=1;smr=1"
  say "and pass it as NINFER_PROBE_KEY=... Inventing a key from the build arch would be a"
  say "declaration wearing a measurement's name, which is what this rewrite exists to stop."
  exit 8
fi

# ---------------------------------------------------------------------------
# THE SIMULATOR GUARD -- a probe run under a SIMULATED rung may not write a MEASURED list.
#
# WHY. The test-only architecture simulator (src/core/arch_sim.h; NINFER_SIM_ARCH +
# NINFER_SIM_ARCH_ACK + the NINFER_ENABLE_ARCH_SIM build key) changes which ROUTE the tables pick.
# The cubins are still this binary's -- but the routing an arm runs under is a rung this box is
# not, so a row written then would record THIS card's numbers under ANOTHER rung's routing: a
# declaration wearing a measurement's name, which is the one thing this list may not contain (and
# the reason the key is the measured capability vector rather than a name).
#
# THE SIM STATE IS RECORDED IN THE RAW LOG as well as enforced here, so `--derive-only` cannot
# lose it: a guard that lived only in the run branch would be bypassed by exactly that pass, and
# the raw log is what a later derivation reads.
#
# IT MARKS AND REFUSES, IT DOES NOT REMOVE. With NINFER_PROBE_ALLOW_SIM_ROUTE=1 the run proceeds,
# the list is written with meta source="simulated_route", and EVERY row becomes
# leg=route_simulated supported=no simulated=yes sim_rung=<rung> -- so the measured numbers and
# cases are kept, and src/core/format_probe.h refuses that shape as support by construction. The
# simulator is the only instrument that exercises the lower rungs at all (the CUDA 12.8 chain at
# /mnt/g/cuda12/tk reaches compute_50/52/53 -- Maxwell, the GTX 960 -- and compute_60/61/62,
# Pascal), so it is marked and refused rather than deleted.
# ---------------------------------------------------------------------------
SIM_ARCH_ENV="${NINFER_SIM_ARCH:-}"
SIM_ACK_ENV="${NINFER_SIM_ARCH_ACK:-}"
SIM_STATE="none"
SIM_RUNG=""
if [ -n "$SIM_ARCH_ENV" ]; then
  SIM_STATE="active:${SIM_ARCH_ENV}"
  SIM_RUNG="$SIM_ARCH_ENV"
elif [ -n "$SIM_ACK_ENV" ]; then
  SIM_STATE="ack-only"
fi
# ---------------------------------------------------------------------------
# THE RAW'S OWN PROVENANCE, RESOLVED FROM THE ARMS AND FROM THE BANNER.
#
# A RAW LOG IS AN APPEND-ONLY LOG OF *MANY RUNS*, SO ITS PROVENANCE IS NOT ONE VALUE. The
# `# probe_sim=` line above is written only when the file is CREATED, so after a second
# invocation appends to the same raw it is STALE, and a resolver that trusts it is answered
# by a line that describes a run other than the one that produced the numbers being derived.
# MEASURED 2026-09-18 (SIMFIX arm 1): a raw created by a real run (`# probe_sim=none`) then
# appended to by an NINFER_PROBE_ALLOW_SIM_ROUTE=1 run derived meta source="real_gpu" with
# supported=yes rows harvested from the SIMULATED arm, and format_probe.h ADMITTED them.
#
# So the resolution below reads the ARMS, and the banner test is UNCONDITIONAL rather than an
# `elif`: no record -- file-level or per-arm -- can mask a banner that is present in the log.
# Precedence, strongest first:
#   1. a `SIMULATED ARCHITECTURE` banner ANYWHERE in the raw  -> simulated (rung unrecorded)
#   2. any arm header recording a sim state other than `none` -> simulated, that rung
#   3. any arm header with NO sim record (a raw from before per-arm records existed) -> the
#      banner in (1) is the record, because the only writer of such a raw invoked every arm
#      as `out="$(NINFER_OP_REPORT_STATS=1 "$bin" 2>&1)"` and so could not have discarded it.
#      MEASURED: /home/user/scratch/BACKUP-SIMPROBE/probe_formats.sh.pre (3f034a31c6c5699e)
#      line 172 has the `2>&1`. A legacy raw with no banner is therefore read as real, and
#      every raw written from now on carries a per-arm record so no new legacy raw appears.
#   4. every arm header recording `sim=none`                     -> real (a positive record)
# ---------------------------------------------------------------------------
RAW_SIM_DEC="none"; RAW_SIM_VALUE="none"; RAW_SIM_RUNG=""; RAW_SIM_ARMS=""; RAW_SIM_WHY=""
resolve_raw_sim() {
  RAW_SIM_DEC="none"; RAW_SIM_VALUE="none"; RAW_SIM_RUNG=""; RAW_SIM_ARMS=""; RAW_SIM_WHY=""
  [ -f "$RAW" ] || return 0
  local prov dec simval rung arms
  prov="$(awk '
    /^##### ARM / {
      fmt = $3; p = "?"; s = "unrecorded"
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^path=/) p = substr($i, 6)
        if ($i ~ /^sim=/)  s = substr($i, 5)
      }
      arms = arms (arms == "" ? "" : "+") fmt "/" p "=" s
      n++
      if (s != "none" && s != "unrecorded") {
        nonsim++
        if (simval == "") { simval = s; r = s; sub(/^active:/, "", r) }
      }
      if (s == "unrecorded") unrec++
    }
    END {
      if (n == 0)            { print "none|none||" arms; exit }
      if (nonsim > 0)        { print "simulated|" simval "|" r "|" arms; exit }
      if (unrec > 0)         { print "legacy|none||" arms; exit }
      print "none|none||" arms
    }' "$RAW")"
  IFS='|' read -r dec simval rung arms <<<"$prov"
  RAW_SIM_DEC="$dec"; RAW_SIM_VALUE="$simval"; RAW_SIM_RUNG="$rung"; RAW_SIM_ARMS="$arms"
  if grep -q 'SIMULATED ARCHITECTURE' "$RAW"; then
    # THE BANNER DECIDES THE CLASS; THE RECORDS STILL SUPPLY THE RUNG WHEN THEY HAVE IT. A
    # refusal that says "the rung was not recorded" while the per-arm records carry
    # `sim=active:70` understates the evidence in hand, and the rung is what makes the refusal
    # re-probeable on the named card.
    RAW_SIM_DEC="simulated"
    RAW_SIM_WHY="banner-in-raw"
    if [ -n "$simval" ] && [ "$simval" != "none" ]; then
      RAW_SIM_VALUE="$simval"; RAW_SIM_RUNG="$rung"
    else
      RAW_SIM_VALUE="active:unrecorded"; RAW_SIM_RUNG=""
    fi
    return 0
  fi
  case "$dec" in
    simulated) RAW_SIM_WHY="per-arm-record" ;;
    legacy)    RAW_SIM_WHY="legacy-no-record-no-banner" ;;
    *)         RAW_SIM_WHY="per-arm-record-all-none" ;;
  esac
}

resolve_raw_sim
# The live environment is still authoritative for the run about to happen, and a --derive-only
# pass has no run; in BOTH cases a simulated class, from either source, makes the list simulated.
if [ "$DERIVE_ONLY" = "1" ] && [ "$RAW_SIM_DEC" = "simulated" ]; then
  SIM_STATE="$RAW_SIM_VALUE"
  SIM_RUNG="$RAW_SIM_RUNG"
fi
RAW_CLASS="empty"
if [ -f "$RAW" ] && [ "$(grep -c '^##### ARM ' "$RAW" || true)" != "0" ]; then
  case "$RAW_SIM_DEC" in
    simulated) RAW_CLASS="simulated" ;;
    legacy)    RAW_CLASS="legacy" ;;
    *)         RAW_CLASS="real" ;;
  esac
fi
# A live run may not append to a raw of a different provenance class: THAT is what leaves a
# stale file-level record behind. Refuse and name the on-ramp instead of writing a mixed log.
MY_CLASS="real"
[ "$SIM_STATE" != "none" ] && MY_CLASS="simulated"
LAUNDERS="no"
[ "$DERIVE_ONLY" = "0" ] && [ "$MY_CLASS" = "simulated" ] && [ -n "$RAW_CLASS" ] && \
  [ "$RAW_CLASS" != "empty" ] && [ "$RAW_CLASS" != "simulated" ] && LAUNDERS="yes"
if [ "$LAUNDERS" = "yes" ]; then
  say "REFUSING TO APPEND TO $RAW: that raw's arms are class '$RAW_CLASS', this run is class"
  say "'$MY_CLASS'. A raw log is an append-only record of MANY runs, and its provenance is not"
  say "one value: appending arms of one class to a raw of another leaves a file-level record"
  say "(`# probe_sim=` is written only at creation) that describes a run other than the one"
  say "that produced some of its arms -- and a derivation that trusted it would write a"
  say "REAL_GPU list out of simulated rows. MEASURED, 2026-09-18: exactly that list was"
  say "ADMITTED as support by src/core/format_probe.h."
  say "ON-RAMP: a fresh raw per provenance class -- point OUT at a new path, e.g."
  say "    tools/archkit/probe_formats.sh '$ARCH' '<new>.list'"
  say "(arms are resumable per raw, so a new raw costs a full re-run of the arms)."
  say "Nothing was written to $OUT."
  exit 11
fi
if [ "$SIM_STATE" = "none" ] && [ "$RAW_SIM_DEC" = "simulated" ]; then
  SIM_STATE="$RAW_SIM_VALUE"
  SIM_RUNG="$RAW_SIM_RUNG"
fi
# NOTE on --derive-only: it appends NO arm, so there is nothing for it to mix, and it is NOT
# refused above. Its answer comes from the raw's own provenance (or from the environment, which
# is the stricter of the two), it is written MARKED and refused by the gate, and the measured
# rows are kept -- mark-and-refuse is this project's doctrine for a simulated route, and a
# read-only pass should not fail for a write-side hazard.
# The other direction is allowed because it cannot launder -- the per-arm union resolves the
# whole raw to `simulated`, so every row is written marked and the gate refuses the list -- but
# it must not be silent: the real arms' support is withheld because the raw is not clean.
if [ "$MY_CLASS" = "real" ] && [ "$RAW_CLASS" = "simulated" ]; then
  say "NOTE: $RAW carries simulated arms; this real run appends to them, and the per-arm"
  say "provenance union makes the WHOLE derived list simulated_route, so this run's real rows"
  say "will be REFUSED as support along with the simulated ones. For a usable list, point OUT"
  say "at a path whose raw holds only real arms."
fi
SOURCE="real_gpu"
[ "$SIM_STATE" != "none" ] && SOURCE="simulated_route"
if [ "$SIM_STATE" != "none" ]; then
  if [ "$DERIVE_ONLY" = "0" ] && [ "${NINFER_PROBE_ALLOW_SIM_ROUTE:-0}" != "1" ]; then
    say "REFUSING TO PROBE: a simulated architecture is active ($SIM_STATE)."
    say "The arms would run under a ROUTE chosen for a rung this box is not, so their rows would"
    say "be about this card's numbers under another rung's routing -- a declaration wearing a"
    say "measurement's name. A route verdict is not a capability verdict, and it is not support."
    say "ON-RAMP: unset the simulator and re-run; that is the only way to obtain a real row:"
    say "    unset NINFER_SIM_ARCH NINFER_SIM_ARCH_ACK"
    say "    NINFER_PROBE_KEY='${PROBE_KEY}' NINFER_PROBE_DECLARED='${DECLARED}' \\"
    say "      tools/archkit/probe_formats.sh '$OUT'"
    say "To record a simulated run instead -- it can never be support, but it does record that the"
    say "path was EXERCISED on that rung -- re-run with NINFER_PROBE_ALLOW_SIM_ROUTE=1."
    say "Nothing was written to $OUT."
    exit 10
  fi
  say "SIMULATED ROUTE RUN ($SIM_STATE): writing meta source=\"simulated_route\" and marking every"
  say "row leg=route_simulated supported=no simulated=yes sim_rung=\"${SIM_RUNG}\"; such a list is"
  say "REFUSED as support by src/core/format_probe.h and records only that the path was taken."
fi

if [ "$DERIVE_ONLY" = "0" ]; then
  # -------------------------------------------------------------------------
  # 0. Take the GPU lock. Everything below is a GPU arm.
  #
  # THE WAIT IS BOUNDED AND OVERRIDABLE. An attempt that cannot get the lock in
  # NINFER_PROBE_LOCK_WAIT_S reports that as its result instead of stalling: the GPU is a
  # shared resource and "I waited forever" is not a measurement.
  # -------------------------------------------------------------------------
  say "== probe_formats.sh arch=$ARCH out=$OUT $(date '+%F %T') =="
  IDLE_W="${NINFER_PROBE_IDLE_WAIT_S:-1800}"
  LOCK_W="${NINFER_PROBE_LOCK_WAIT_S:-3600}"
  export NINFER_LOCK_MIN_AVAIL_MB="${NINFER_LOCK_MIN_AVAIL_MB:-19500}"
  # shellcheck disable=SC1091
  source /mnt/c/Users/User/Documents/ziqinzhang/sh/_lock.sh gpu "$IDLE_W" "$LOCK_W"
  LRC=$?
  if [ "$LRC" != "0" ]; then
    say "NO GPU LOCK (rc=$LRC, idle_wait=${IDLE_W}s lock_wait=${LOCK_W}s): refusing to run any"
    say "arm. A probe verdict produced outside the lock would be a verdict about a contended"
    say "GPU, which is not a property of the kernel. Retry later, or raise"
    say "NINFER_PROBE_LOCK_WAIT_S. Nothing was written to $OUT."
    exit 9
  fi
  trap 'exec 9>&-' EXIT

  say "arch_suffix=$ARCH_SUFFIX lib_sha16=$LIB_SHA host=$HOST_GPU"

  # -------------------------------------------------------------------------
  # 2. Run every arm. RESUMABLE: an arm whose marker is already in $RAW is skipped, and
  # each arm's output is appended as soon as it finishes, so an interrupted invocation
  # never loses an arm that already ran and never claims one that did not.
  # -------------------------------------------------------------------------
  if [ ! -f "$RAW" ]; then
    {
      printf '# probe_formats.sh raw arm log -- appended, never rewritten\n'
      printf '# git HEAD %s\n' "$HEAD_SHA"
      printf '# arch_suffix=%s lib_sha16=%s host=%s\n' "$ARCH_SUFFIX" "$LIB_SHA" "$HOST_GPU"
      printf '# probe_sim=%s\n' "$SIM_STATE"
    } > "$RAW"
  fi
  for row in "${ARMS[@]}"; do
    IFS='|' read -r fmt path target <<<"$row"
    bin="$BIN_DIR/$target"
    if [ ! -x "$bin" ]; then
      say "-- SKIP $fmt/$path: $target absent (arm cannot run: not built)"
      continue
    fi
    sha="$(sha256sum "$bin" | cut -c1-16)"
    if grep -q "^##### ARM $fmt path=$path " "$RAW"; then
      say "-- SKIP $fmt/$path: already measured in $RAW (resume)"
      continue
    fi
    s=$(date +%s)
    out="$(NINFER_OP_REPORT_STATS=1 "$bin" 2>&1)"
    rc=$?
    e=$(date +%s)
    {
      printf '##### ARM %s path=%s target=%s rc=%s sha16=%s elapsed=%ss sim=%s\n' \
        "$fmt" "$path" "$target" "$rc" "$sha" "$((e-s))" "$SIM_STATE"
      printf '%s\n' "$out"
    } >>"$RAW"
    say "-- $fmt/$path $target rc=$rc sha16=$sha elapsed=$((e-s))s"
  done
else
  say "== probe_formats.sh --derive-only arch=$ARCH out=$OUT $(date '+%F %T') =="
  [ -f "$RAW" ] || { say "no raw log at $RAW: nothing to derive"; exit 4; }
fi

# ---------------------------------------------------------------------------
# 3. Derive per-(format, path, band) verdicts from the harvested cases.
#
# A case label looks like:  NVFP4_A4 [14336,5120] T=4
# The band comes from T via the SAME edges src/core/format_probe.h uses (1..3 / 4..8 /
# 9..16 / >=17), so list and gate cannot disagree about a band.
#
# THE VERDICT RULE, stated once and fail-closed:
#   supported=yes  iff  the ARM was green (rc=0)  AND  the band has >= 1 measured case
#                       AND every measured case in the band passed
#                       (rel_l2 <= rel_l2_limit, gross_ratio <= 1, non_finite == 0)
#   supported=no   otherwise, with arm_rc / fail / cases / ratio recorded so the reason is
#                  diagnosable. A band with zero cases is written leg=unprobed cases=0 and
#                  the gate REFUSES it.
# ---------------------------------------------------------------------------
ROWS="$(awk '
function band_of(t) {
  if (t >= 1  && t <= 3)  return "1..3"
  if (t >= 4  && t <= 8)  return "4..8"
  if (t >= 9  && t <= 16) return "9..16"
  if (t >= 17)            return "17..open"
  return ""
}
/^##### ARM / {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^path=/)   p = substr($i, 6)
    if ($i ~ /^rc=/)     rc = substr($i, 4)
    if ($i ~ /^sha16=/)  sha = substr($i, 7)
  }
  fmt = $3
  pk = fmt "/" p
  armrc[pk] = rc
  armsha[pk] = sha
  if (rc != 0) badarm[pk] = 1
  seenpair[pk] = 1
  next
}
/OP_ERROR_STATS kind=reduction/ {
  rel = rel_lim = gr = nf = -1
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^rel_l2=/)        rel     = substr($i, 8)  + 0
    if ($i ~ /^rel_l2_limit=/)  rel_lim = substr($i, 14) + 0
    if ($i ~ /^gross_ratio=/)   gr      = substr($i, 13) + 0
    if ($i ~ /^non_finite=/)    nf      = substr($i, 12) + 0
  }
  ci = index($0, "case=")
  if (ci <= 0) next
  label = substr($0, ci + 5)
  gsub(/^"|"$/, "", label)
  t = -1
  if (match(label, /T=[0-9]+/)) t = substr(label, RSTART + 2, RLENGTH - 2) + 0
  b = band_of(t)
  if (b == "") next
  key = pk "/" b
  seen[key]++
  ratio = (rel_lim > 0 ? rel / rel_lim : (rel > 0 ? 99 : 0))
  if (gr > ratio) ratio = gr
  casefail = (rel_lim >= 0 && rel > rel_lim) || (gr > 1.0) || (nf != 0)
  if (casefail) {
    bad[key]++
    if (!(key in firstfail)) firstfail[key] = label
  }
  if (!(key in worst) || ratio > worst[key]) worst[key] = ratio
  if (!(key in example)) example[key] = label
  geom = label; sub(/^[^[]*\[/, "", geom); sub(/\].*$/, "", geom)
  if (geom != "") geoms[key "/" geom] = 1
}
END {
  nb = split("1..3 4..8 9..16 17..open", bands, " ")
  for (pk in seenpair) {
    for (i = 1; i <= nb; i++) {
      k = pk "/" bands[i]
      n = seen[k] + 0
      nbad = bad[k] + 0
      ng = 0
      for (g in geoms) { if (index(g, pk "/" bands[i] "/") == 1) ng++ }
      sup = (n > 0 && nbad == 0 && armrc[pk] == 0) ? "yes" : "no"
      leg = (n > 0) ? "reduction_l2" : "unprobed"
      w = (k in worst) ? worst[k] : 0
      ff = (k in firstfail) ? firstfail[k] : ""
      printf "%s|%s|%s|%s|%d|%.6g|%s|%s|%s|%d|%d\n", pk, bands[i], sup, leg, n, w,
             (k in example ? example[k] : ""), armsha[pk], ff, armrc[pk], ng
    }
  }
}' "$RAW" | sort -t'|' -k1,1 -k2,2)"

if [ -z "$ROWS" ]; then
  say "NO ROWS derived: no arm marker was found in $RAW. Writing NOTHING."
  say "An empty harvest is not an empty support list; it is a probe that did not measure."
  exit 4
fi

# ---------------------------------------------------------------------------
# 4. Emit the list.
# ---------------------------------------------------------------------------
FAILING="$(grep '^##### ARM ' "$RAW" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^rc=/ && $i != "rc=0") print $3"/"$4"("$i")"}' | tr '\n' ' ')"
{
  printf '# ninfer format support list v2 -- PRODUCED BY A PROBE RUN, NEVER HAND-MAINTAINED.\n'
  printf '# Written by tools/archkit/probe_formats.sh at %s. git HEAD %s.\n' "$(date '+%F %T')" "$HEAD_SHA"
  printf 'meta probe_tool="tools/archkit/probe_formats.sh"\n'
  printf 'meta criterion_id="linear-reduction-v1"\n'
  printf 'meta criterion_src="tests/ops/linear/linear_test_common.cpp:38-48"\n'
  printf 'meta reference="cpu_linear_gemm_fp64 (tests/ops/linear/linear_test_common.cpp:246)"\n'
  printf 'meta stats_env="NINFER_OP_REPORT_STATS=1"\n'
  printf 'meta key="%s"\n' "$PROBE_KEY"
  printf 'meta declared_name="%s"\n' "$DECLARED_NAME"
  printf 'meta declared_cc="%s"\n' "$DECLARED_CC"
  printf 'meta declared_build="%s"\n' "${DECLARED_BUILD:-$ARCH_SUFFIX}"
  printf 'meta date="%s"\n' "$(date '+%F')"
  printf 'meta binary_path="build/tests (per arm; one target per format+path)"\n'
  printf 'meta binary_sha16="%s"\n' "$(sed -n 's/.* sha16=\([0-9a-f]*\).*/\1/p' "$RAW" | sort -u | head -1)"
  printf 'meta binary_mtime="%s"\n' "$(date '+%F %T')"
  printf 'meta library_path="build/src/libninfer_ops.a"\n'
  printf 'meta library_sha16="%s"\n' "$LIB_SHA"
  printf 'meta library_mtime="%s"\n' "$LIB_MTIME"
  printf 'meta source="%s"\n' "$SOURCE"
  [ "$SIM_STATE" != "none" ] && printf 'meta sim_run_state="%s"\n' "$SIM_STATE"
  [ -n "$SIM_RUNG" ] && printf 'meta sim_run_rung="%s"\n' "$SIM_RUNG"
  # PER-ARM PROVENANCE. Every arm that produced a number in this list is named with the class
  # it was measured under, so a raw that mixes classes is visible in the list itself and not
  # only in the log. A reader who has the list and not the raw can still see the mixture.
  [ -n "$RAW_SIM_ARMS" ] && printf 'meta sim_arms="%s"\n' "$RAW_SIM_ARMS"
  [ -n "$RAW_SIM_WHY" ] && printf 'meta provenance_resolved_by="%s"\n' "$RAW_SIM_WHY"
  printf 'meta arms_not_green="%s"\n' "${FAILING:-none}"
  printf '#\n'
  printf '# A band with cases=0 and leg=unprobed was NEVER MEASURED: the gate REFUSES it.\n'
  printf '# supported=yes requires the arm to be green AND every measured case in the band\n'
  printf '# to pass the criterion. A band with fail= names the first case that did not.\n'
  printf '#\n'
  printf '%s\n' "$ROWS" | while IFS='|' read -r pk band sup leg n w example sha ff armrc ng; do
    fmt="${pk%%/*}"
    path="${pk##*/}"
    # MARK AND REFUSE: under a simulated route no row may claim arithmetic support, and the rung
    # that was simulated is recorded so the refusal can name it. The measured cases/ratio are kept.
    if [ "$SOURCE" = "simulated_route" ]; then
      sup="no"
      leg="route_simulated"
    fi
    printf 'entry key="%s" declared_name="%s" declared_cc="%s" format=%s path=%s band=%s supported=%s leg=%s cases=%s geometries=%s arm_rc=%s' \
      "$PROBE_KEY" "$DECLARED_NAME" "$DECLARED_CC" "$fmt" "$path" "$band" "$sup" "$leg" "$n" "$ng" "$armrc"
    [ -n "$example" ] && printf ' ratio=%s case="%s"' "$w" "$example"
    [ -n "$ff" ] && printf ' fail="%s"' "$ff"
    [ "$SOURCE" = "simulated_route" ] && printf ' simulated=yes'
    [ "$SOURCE" = "simulated_route" ] && [ -n "$SIM_RUNG" ] && printf ' sim_rung=%s' "$SIM_RUNG"
    printf ' probe="build/tests sha16=%s"\n' "$sha"
  done
} > "$OUT"

say "wrote $OUT ($(grep -c '^entry ' "$OUT") entries, $(grep -c '^entry .*supported=yes' "$OUT") supported)"
say "arms not green: ${FAILING:-none}"
say "raw arm output: $RAW"
say "== probe_formats.sh end $(date '+%F %T') =="
