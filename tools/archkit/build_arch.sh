#!/usr/bin/env bash
# build_arch.sh -- multi-GPU build matrix helper (CPU-only compilation; run on
# the target toolchain, no target GPU required).
#
# Usage:
#   ./build_arch.sh 86            # one arch -> dist-86/ninfer-serve
#   ./build_arch.sh 75 86 89 120a # several archs, one dist dir each
#   ./build_arch.sh --rung legacy # the whole pre-75 block: 50 52 53 60 61 62 70
#   ./build_arch.sh --list-rungs  # what the sets are, and which toolkit each needs
#
# Each arch gets its own build dir AND its own dist dir, so binaries can be
# shipped per GPU tier (see tools/archkit/_GPU_MATRIX.md). Defaults to Release + apps.
#
# PER-TIER TOOLCHAIN (MULTIARCH). "Which archs does one toolkit cover?" is a
# property of the TOOLKIT, not of this project. Measured on this box:
#   * CUDA 13.1/13.3 covers 75..121 (with the '-a' and '-f' suffixes) and does
#     NOT cover 70 at all -- `nvcc --list-gpu-arch` has no compute_70 and
#     `nvcc -arch=sm_70` exits 1 with
#       "nvcc fatal : Unsupported gpu architecture 'sm_70'".
#   * CUDA 12.8 (the root the legacy toolchain file pins) answers
#       compute_50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120
#     so ONE toolkit covers the whole PRE-75 BLOCK -- Maxwell (50/52/53),
#     Pascal (60/61/62) and Volta (70) -- and that block is therefore ONE tier:
#     one toolkit, one toolchain file, one dist per rung. That is the set
#     LEGACY_ARCHS below, and it is the ONLY place that set is written down
#     (TOOLCHAIN_FILE_OF is generated from it, and the drift check refuses a
#     legacy rung with no entry).
#
#   NINFER_CUDA_TOOLKIT        default toolkit root (e.g. /usr/local/cuda-13.3)
#   NINFER_CUDA_TOOLKIT_<arch> per-arch override (e.g. NINFER_CUDA_TOOLKIT_70)
#
# A pre-13 toolkit is necessary but NOT sufficient for the legacy rungs, which is
# why they go through tools/archkit/cuda128_legacy_toolchain.cmake instead of a
# bare -DCMAKE_CUDA_COMPILER. On this distro CUDA 12.8 refuses the HOST image in
# two independent places and neither is about the rung: crt/host_config.h:143
# (`__GNUC__ > 14`, system g++ is 15.2) and the cospi/sinpi/rsqrt exception-spec
# clash between crt/math_functions.h and glibc 2.43's bits/mathcalls.h. The
# toolchain file pins g++-13 and puts a matching glibc-2.35 header set in front.
# Measured; see tools/archkit/cuda128_legacy_toolchain.cmake for the full note.
# (The 70-only predecessor of that file, cuda128_sm70_toolchain.cmake, is still
# on disk and is still the measured record the sm_70 report and AGENTS.md cite;
# it is the same host recipe. New rungs go in the general file, not a third one.)
#
# REFUSED ON PURPOSE: one invocation spanning two tiers. Two cubins from two
# cudart versions cannot be linked into one binary, so a list that needs two
# toolkits is a request for two dists -- run the script twice. The tier of an
# arch is: its TOOLCHAIN FILE if it has one (the file decides the compiler),
# else its NINFER_CUDA_TOOLKIT[_<arch>] compiler, else "whatever cmake finds".
# MEASURED DEFECT THIS CLOSES: grouping by compiler alone reported the whole
# legacy block as TWO tiers whenever only the documented single-rung override
# (NINFER_CUDA_TOOLKIT_70=/mnt/g/cuda12/tk, see AGENTS.md) was set -- arch 70
# resolved to that nvcc and arch 52 to the empty string -- so the one invocation
# this tier exists for was refused. It also accepted `build_arch.sh 70 86` with
# no override set, which is exactly the two-toolkit list it claims to refuse.
set -euo pipefail
# REPO ROOT. MEASURED DEFECT THIS FIXES: this line used to read
#   cd "$(dirname "$0")/.."
# which lands in <repo>/tools, NOT the repo root -- dirname of
# tools/archkit/build_arch.sh is tools/archkit, so one `..` is `tools/`. So
# every `cmake -S .` below ran in a directory with no CMakeLists.txt, i.e. this
# script could never configure anything, for any arch, including the 70 rung it
# was extended for. Evidence: <repo>/tools has no CMakeLists.txt (exit 1) while
# the repo root does; build-sm70/CMakeCache.txt records
# CMAKE_HOME_DIRECTORY=/home/user/ninfer-fusion, i.e. that configure was run by
# hand from the repo root and not through this script (which is also why
# build-sm70 held 2 objects and no dist-70 existed). The `..` count is two, and
# readlink -f means an invocation through a symlinked path resolves the same
# way. The sentinel below is what keeps a future edit from silently re-breaking
# it: a wrong cwd must fail here, loudly, and not inside cmake.
cd "$(dirname "$(readlink -f "$0")")/../.."
if [ ! -f CMakeLists.txt ]; then
  echo "build_arch.sh: after cd, $PWD has no CMakeLists.txt -- this script must live in <repo>/tools/archkit/" >&2
  exit 6
fi

# ---------------------------------------------------------------------------
# THE PRE-75 RUNGS, and the one toolchain file that builds all of them.
#
# This array IS the selector: TOOLCHAIN_FILE_OF below is generated from it, so
# "which rungs need a pre-13 toolkit" is answered in one place and cannot drift
# from the table that implements it. Adding a rung here is also what makes the
# drift check demand an entry for it.
readonly LEGACY_ARCHS=(50 52 53 60 61 62 70)
readonly LEGACY_TOOLCHAIN="tools/archkit/cuda128_legacy_toolchain.cmake"
# The toolkit root that file pins, repeated here for ONE purpose: the CMake arch
# gate at the top of CMakeLists.txt probes an nvcc BEFORE project() reads any
# toolchain file, so without a compiler path in the cache it cannot answer and
# falls back to its historical floor of 70 -- which refuses every pre-70 rung.
# MEASURED: `... build_arch.sh 50` with the file above configured, then died with
#   "NInfer: arch 50 is below the floor this project has ever measured ...
#    the `nvcc --list-gpu-arch` probe did not answer (rc=n/a: the compiler was not found)"
# so the gate has to be told which nvcc to ask. It must be the SAME toolkit the
# toolchain file uses; the file itself remains the authority on the compiler it
# sets (a normal set() there shadows a -D cache entry), so a disagreement between
# this root and NINFER_CUDA128_ROOT is warned about below rather than obeyed.
readonly LEGACY_TOOLKIT_ROOT_DEFAULT="/mnt/g/cuda12/tk"

# The rung that PROVED the mechanism, kept as a named historical marker and NOT
# as the selector. It is a single number because it was written when the
# question was "does a second toolkit work at all"; the answer is LE the whole
# pre-75 block (see above). src/core/format_probe.h:136 cites this line by name
# as the example of a constant that no code path reads -- it is left here, at
# this line, for that citation to keep resolving, and the fact it used to stand
# for is now carried by LEGACY_ARCHS.
readonly NINFER_ARCH_NEEDING_LEGACY_TOOLKIT=70

# Named rung SETS (mutually combinable with explicit arch numbers; deduplicated
# in the order given). A set exists so the matrix can say "the legacy block"
# without seven numbers, and so the toolkit each set needs is stated next to it.
rung_archs() {
  case "$1" in
    legacy) printf '%s\n' "${LEGACY_ARCHS[@]}" ;;
    *) return 1 ;;
  esac
}
rung_note() {
  case "$1" in
    legacy) printf 'pre-75 block (Maxwell/Pascal/Volta) -- needs %s' "$LEGACY_TOOLCHAIN" ;;
  esac
}
list_rungs() {
  printf 'rungs (sets of archs this script can build together in one invocation):\n'
  local r
  for r in legacy; do
    printf '  %-8s -> %-26s %s\n' "$r" "$(rung_archs "$r" | tr '\n' ' ')" "$(rung_note "$r")"
  done
  printf '\nindividual archs: pass the number directly, e.g. `%s 86 89 120a`.\n' "$0"
  printf 'anything NOT in LEGACY_ARCHS is built by the DEFAULT (CUDA 13.x) toolkit.\n'
}

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  printf '\narchs needing a pre-13 toolkit: %s\n' "${LEGACY_ARCHS[*]}"
}

ARCHS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rung)
      if [ "$#" -lt 2 ]; then echo "build_arch.sh: --rung needs a name (try --list-rungs)" >&2; exit 2; fi
      if ! rung_archs "$2" >/dev/null; then echo "build_arch.sh: unknown rung '$2' (try --list-rungs)" >&2; exit 2; fi
      # shellcheck disable=SC2207
      ARCHS+=($(rung_archs "$2"))
      shift
      ;;
    --rung=*)
      r="${1#--rung=}"
      if ! rung_archs "$r" >/dev/null; then echo "build_arch.sh: unknown rung '$r' (try --list-rungs)" >&2; exit 2; fi
      # shellcheck disable=SC2207
      ARCHS+=($(rung_archs "$r"))
      ;;
    --list-rungs) list_rungs; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "build_arch.sh: unknown option '$1' (try --help)" >&2; exit 2 ;;
    *) ARCHS+=("$1") ;;
  esac
  shift
done

# dedupe, keep first-seen order
_deduped=()
for a in ${ARCHS[@]+"${ARCHS[@]}"}; do
  seen=""
  for b in ${_deduped[@]+"${_deduped[@]}"}; do [ "$b" = "$a" ] && seen=1 && break; done
  [ -n "$seen" ] || _deduped+=("$a")
done
ARCHS=(${_deduped[@]+"${_deduped[@]}"})

if [ "${#ARCHS[@]}" -eq 0 ]; then
  echo "usage: $0 <arch> [arch...] | --rung legacy   e.g. $0 86 89 120a" >&2
  exit 2
fi

# Name -> toolchain file. A legacy rung is not a "-DCMAKE_CUDA_COMPILER" away from
# working: the toolkit it needs also has to agree with the host about gcc and
# about glibc. Declared as data so the reason lives in one place, and GENERATED
# from LEGACY_ARCHS so that set and table cannot disagree.
declare -A TOOLCHAIN_FILE_OF=()
for arch in "${LEGACY_ARCHS[@]}"; do
  TOOLCHAIN_FILE_OF["$arch"]="$LEGACY_TOOLCHAIN"
done

toolkit_for() { # toolkit_for <arch> -> compiler path on stdout
  local arch="$1" var="NINFER_CUDA_TOOLKIT_${arch}"
  local root="${!var:-${NINFER_CUDA_TOOLKIT:-}}"
  if [ -n "$root" ]; then printf '%s/bin/nvcc' "$root"; return; fi
  # A legacy rung falls back to the root its toolchain file pins (see the note on
  # LEGACY_TOOLKIT_ROOT_DEFAULT): the gate above needs an nvcc to ask, and
  # "whatever cmake finds" is a CUDA 13 nvcc that will answer "no compute_52" and
  # turn a supported build into a refused one.
  local l is_legacy=""
  for l in "${LEGACY_ARCHS[@]}"; do [ "$l" = "$arch" ] && is_legacy=1 && break; done
  if [ -n "$is_legacy" ]; then printf '%s/bin/nvcc' "$LEGACY_TOOLKIT_ROOT_DEFAULT"; return; fi
  printf ''
}

# The drift check: a legacy rung must have a toolchain file, and a rung that is
# NOT legacy must not have one (a stale entry would silently build a modern rung
# with a pre-13 toolkit).
for arch in "${ARCHS[@]}"; do
  tcf="${TOOLCHAIN_FILE_OF[$arch]:-}"
  is_legacy=""
  for l in "${LEGACY_ARCHS[@]}"; do [ "$l" = "$arch" ] && is_legacy=1 && break; done
  if [ -n "$is_legacy" ] && [ -z "$tcf" ]; then
    echo "build_arch.sh: arch $arch is in LEGACY_ARCHS but has no TOOLCHAIN_FILE_OF entry." >&2
    exit 5
  fi
  if [ -z "$is_legacy" ] && [ -n "$tcf" ]; then
    echo "build_arch.sh: arch $arch is not in LEGACY_ARCHS but has toolchain file '$tcf'." >&2
    exit 5
  fi
  if [ -n "$tcf" ] && [ ! -f "$tcf" ]; then
    echo "build_arch.sh: toolchain file for arch $arch does not exist: $tcf" >&2
    exit 5
  fi
done

declare -A TOOLKIT_OF=()
declare -A TIER_OF=()
for arch in "${ARCHS[@]}"; do
  TOOLKIT_OF["$arch"]="$(toolkit_for "$arch")"
  tcf="${TOOLCHAIN_FILE_OF[$arch]:-}"
  if [ -n "$tcf" ]; then
    # The toolchain file sets CMAKE_CUDA_COMPILER itself, so the file -- not the
    # environment -- is what identifies the compiler for this rung.
    TIER_OF["$arch"]="file:$tcf"
    if [ -n "${TOOLKIT_OF[$arch]}" ] && [ "${TOOLKIT_OF[$arch]}" != "${LEGACY_TOOLKIT_ROOT_DEFAULT}/bin/nvcc" ]; then
      echo "build_arch.sh: NOTE arch $arch -- the toolchain file chooses the compiler (its own" >&2
      echo "  NINFER_CUDA128_ROOT, default ${LEGACY_TOOLKIT_ROOT_DEFAULT}); the path seen here" >&2
      echo "  (${TOOLKIT_OF[$arch]}) is only the nvcc the CMake arch gate is asked to probe," >&2
      echo "  and the toolchain file wins. If you meant to move the toolkit, pass" >&2
      echo "  -DNINFER_CUDA128_ROOT=<root> to cmake, or edit the file's default." >&2
    fi
  elif [ -n "${TOOLKIT_OF[$arch]}" ]; then
    TIER_OF["$arch"]="compiler:${TOOLKIT_OF[$arch]}"
  else
    TIER_OF["$arch"]="cmake-found"
  fi
done

distinct="$(for a in "${ARCHS[@]}"; do printf '%s\n' "${TIER_OF[$a]}"; done | sort -u | wc -l)"
if [ "$distinct" -gt 1 ]; then
  echo "build_arch.sh: this arch list spans $distinct tiers:" >&2
  for a in "${ARCHS[@]}"; do
    echo "  $a -> ${TIER_OF[$a]}${TOOLKIT_OF[$a]:+ (compiler ${TOOLKIT_OF[$a]})}" >&2
  done
  echo "Two cubins built by two cudart versions cannot go into one binary." >&2
  echo "Run the script once per tier, so each tier gets its own dist. The rung" >&2
  echo "sets are listed by --list-rungs." >&2
  exit 3
fi

GEN="${NINFER_CMAKE_GENERATOR:-Ninja}"
for arch in "${ARCHS[@]}"; do
  bdir="build-${arch}"
  ddir="dist-${arch}"
  tc="${TOOLKIT_OF[$arch]}"
  echo "== configuring ${bdir} (arch ${arch}${tc:+, compiler ${tc}})"
  tcf="${TOOLCHAIN_FILE_OF[$arch]:-}"
  if [ -n "$tcf" ]; then
    # -DCMAKE_CUDA_ARCHITECTURES is NOT redundant here even though the toolchain
    # file also handles the arch: CMakeLists.txt:20-21 sets a 120a default BEFORE
    # project() reads the toolchain file, and a CACHE set without FORCE cannot
    # displace an existing entry -- so without this the legacy build configures
    # as 120a and the tree (correctly) refuses it on CUDA 12.8. It is also what
    # tells the ONE legacy toolchain file WHICH rung this dist is (the file no
    # longer hardcodes 70), so the rung set and the toolchain file cannot drift.
    echo "   toolchain ${tcf}"
    cmake -S . -B "${bdir}" -G "${GEN}" \
      -DCMAKE_TOOLCHAIN_FILE="${tcf}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="${tc}" \
      -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
  elif [ -n "$tc" ]; then
    cmake -S . -B "${bdir}" -G "${GEN}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="${tc}" \
      -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
  else
    cmake -S . -B "${bdir}" -G "${GEN}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
  fi
  echo "== building ${bdir}"
  cmake --build "${bdir}" -j"${NINFER_JOBS:-$(nproc)}"
  echo "== staging ${ddir}"
  mkdir -p "${ddir}"
  if [ -f "${bdir}/apps/ninfer-serve" ]; then
    cp -a "${bdir}/apps/ninfer-serve" "${ddir}/ninfer-serve"
    echo "   ${ddir}/ninfer-serve"
  else
    echo "build_arch.sh: ${bdir}/apps/ninfer-serve was not produced" >&2
    exit 4
  fi
done
echo "matrix build complete"
