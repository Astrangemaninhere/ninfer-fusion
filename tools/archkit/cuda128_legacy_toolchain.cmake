# cuda128_legacy_toolchain.cmake
# ---------------------------------------------------------------------------
# Host+device toolchain for the PRE-75 rungs of ninfer-fusion: the whole
# Maxwell (sm_50/52/53) + Pascal (sm_60/61/62) + Volta (sm_70) block that
# CUDA 13 dropped offline code generation for.
#
# WHY ONE FILE FOR SEVEN RUNGS. "Which rungs does one toolkit cover?" is a
# property of the TOOLKIT, not of the rung. Measured on this box, CUDA 12.8 at
# NINFER_CUDA128_ROOT below (`nvcc --list-gpu-arch`):
#
#     compute_50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120
#
# so ONE toolkit family covers all seven pre-75 rungs, and they are therefore
# ONE tier: one toolkit root, one set of host-image workarounds, seven dists.
# CUDA 13.x lists none of them (`nvcc -arch=sm_70` exits 1: "Unsupported gpu
# architecture"), which is why the same rung cannot be reached from a fat list
# in the default toolkit -- see tools/archkit/build_arch.sh, whose
# LEGACY_ARCHS array is the selector and which points every member here.
#
# WHY THE HOST IMAGE NEEDS FIXING AT ALL (two facts, both measured, neither of
# them about any particular rung, see /home/user/scratch/PATCHSET/CUDA128/REPORT.md):
#
#  1. CUDA 12.8's nvcc refuses this distro's HOST image, in two independent
#     places, and neither of them is about the target:
#       (a) crt/host_config.h:143 -- `__GNUC__ > 14` -- the system g++ is 15.2,
#           so the gate trips before any code is seen.  Fixed by -ccbin/g++-13
#           (g++-13 = 13.4.0, installed, no install performed).
#       (b) crt/math_functions.h:2601/2556/597 declare cospi/sinpi/rsqrt
#           WITHOUT `noexcept`; this distro's glibc 2.43 declares them WITH it
#           (bits/mathcalls.h:83/85/206, gated by
#           `__GLIBC_USE (IEC_60559_FUNCS_EXT_C23)`), so the C++ front end
#           emits 6x "exception specification is incompatible" errors.  Fixed
#           by putting a glibc 2.35 (Ubuntu 22.04) header set in front of the
#           host headers -- which is exactly the glibc CUDA 12.8 was built
#           against.  /mnt/g/cuda12/jammy-sysroot/usr/include is that header
#           set and it is already on this box.
#
# `jammy-sysroot` has NO usr/bin, i.e. it is a HEADER-ONLY image: it can
# redirect <stdio.h>/<bits/*> but it cannot supply a compiler.  That is why
# the host compiler still has to be /usr/bin/g++-13 from the real system.
#
# THIS FILE IS THE GENERALIZATION OF tools/archkit/cuda128_sm70_toolchain.cmake,
# which remains on disk (and is cited as the measured record by the sm_70
# report and by AGENTS.md); the two are the same host recipe. The difference
# is that this file does NOT hardcode the rung: the caller names it with
# -DCMAKE_CUDA_ARCHITECTURES=<rung>, which is also what makes one invocation
# able to build several legacy rungs in sequence. Do not re-fork the host
# workarounds into a third file: fix them here.
#
# Usage (see tools/archkit/build_arch.sh, which passes exactly this):
#   cmake -S . -B build-52 -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=tools/archkit/cuda128_legacy_toolchain.cmake \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=52 \
#         -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
# ---------------------------------------------------------------------------

set(NINFER_CUDA128_ROOT "/mnt/g/cuda12/tk"          CACHE PATH   "CUDA 12.8 toolkit root (the toolkit that still lists compute_50..70)")
set(NINFER_HOST_GCC13     "/usr/bin/g++-13"         CACHE FILEPATH "host C++ for nvcc -ccbin; must be <= gcc 14")
set(NINFER_HOST_GCC13_C   "/usr/bin/gcc-13"         CACHE FILEPATH "host C for nvcc -ccbin")
set(NINFER_JAMMY_HEADERS  "/mnt/g/cuda12/jammy-sysroot/usr/include" CACHE PATH "glibc 2.35 header root that matches CUDA 12.8")

# --- the rung this dist is for ---------------------------------------------
# The set is the fact this file exists for; it must agree with LEGACY_ARCHS in
# tools/archkit/build_arch.sh, and that script REFUSES a legacy rung that has
# no entry in its table, so the two cannot drift apart silently.
set(NINFER_LEGACY_ARCHS "50;52;53;60;61;62;70" CACHE STRING "the pre-75 rungs CUDA 13 can no longer codegen for")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # Reached in two legitimate cases, and in no other: (a) CMake's own
  # try_compile sub-builds do not inherit command-line -D vars, so the
  # compiler-id probe lands here; (b) a caller forgot the -D. The default is
  # only safe for (a) -- for (b) the configure would produce a 70 dist that
  # was asked for as nothing, which is why the warning is printed rather than
  # the value being quietly chosen.
  message(WARNING "cuda128_legacy_toolchain: CMAKE_CUDA_ARCHITECTURES was not set by the caller; "
                  "defaulting to 70 (this default exists for CMake's sub-builds). Pass "
                  "-DCMAKE_CUDA_ARCHITECTURES=<50|52|53|60|61|62|70> for a real dist.")
  set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
endif()

set(_legacy_ok FALSE)
foreach(_legacy_rung IN LISTS NINFER_LEGACY_ARCHS)
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "${_legacy_rung}")
    set(_legacy_ok TRUE)
  endif()
endforeach()
if(NOT _legacy_ok)
  message(FATAL_ERROR
    "cuda128_legacy_toolchain: arch '${CMAKE_CUDA_ARCHITECTURES}' is not one of the pre-75 rungs "
    "this toolkit is for (${NINFER_LEGACY_ARCHS}). Rungs 75 and up are built by the DEFAULT "
    "(CUDA 13.x) toolkit and must not use this file -- see tools/archkit/build_arch.sh.")
endif()

# --- device side -----------------------------------------------------------
set(CMAKE_CUDA_COMPILER      "${NINFER_CUDA128_ROOT}/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "${NINFER_HOST_GCC13}")
set(CMAKE_CUDA_STANDARD 17)

# --- host side -------------------------------------------------------------
# Both compilers are pinned so that nvcc's host passes and the plain C++ TUs
# disagree about nothing; a mismatch here is how "it configured but would not
# link" happens.
set(CMAKE_C_COMPILER   "${NINFER_HOST_GCC13_C}")
set(CMAKE_CXX_COMPILER "${NINFER_HOST_GCC13}")

# --- the glibc header redirection: CUDA compile ONLY, never the host TUs ------
#
# The recorded sm_70 recipe put a glibc-2.35 (Ubuntu 22.04) header set in front of
# the host headers, because CUDA 12.8's crt/math_functions.h declares cospi/sinpi/
# rsqrt WITHOUT `noexcept` while this distro's glibc 2.43 declares them WITH it
# (bits/mathcalls.h, gated by `__GLIBC_USE (IEC_60559_FUNCS_EXT_C23)`), and the
# clash is 6 "exception specification is incompatible" errors. That reason is
# REAL and was re-measured here: without the redirection the CMake compiler-ID
# probe itself dies with exactly those 6 errors in CMakeCUDACompilerId.cu
# (recon/configure_61.log, "6 errors detected in the compilation of
# CMakeCUDACompilerId.cu"), which fails the configure outright.
#
# WHAT THE RECORDED RECIPE GOT WRONG: it applied the same redirection to the
# HOST compilers too (CMAKE_CXX_FLAGS_INIT / CMAKE_C_FLAGS_INIT), and there it is
# not a fix but a break: /mnt/g/cuda12/jammy-sysroot/usr/include is a PARTIAL
# header set (823 .h: stdlib.h and math.h present, sys/cdefs.h ABSENT,
# bits/mathcalls.h only under x86_64-linux-gnu), so putting it in front pairs
# glibc 2.35's features.h with glibc 2.43's stdlib.h and every plain C++ TU dies
# with
#   /usr/include/stdlib.h:752:65: error: expected initializer before '__COLD'
# MEASURED: 15 of 15 host TUs failed this way in the first whole-tree attempt,
# all with that one cause, while the redirection fixes nothing for a TU that does
# not include CUDA's math_functions.h (recon/build_52.ninja_attempt1.out).
#
# SO: -Xcompiler (CUDA compile and the compiler-ID probe) keeps it; the host
# compilers do not get it. NINFER_LEGACY_JAMMY_HEADERS_HOST exists only as an
# escape hatch and should stay OFF -- turning it on brings the __COLD failure back.
#
# -isystem and NOT -idirafter on the CUDA side: -idirafter is searched AFTER the
# standard system dirs, so host mathcalls.h still wins and the 6 errors stay
# (measured, recorded in the sm_70 file). NOTE the token must stay ONE literal
# with commas and no ';' -- a CMake LIST expands to 'a;b', nvcc hands the ';' to
# sh, and the probe then dies with `g++-13: fatal error: no input files`
# (measured). Guarded against a second read of this file so the flags cannot be
# appended twice.
option(NINFER_LEGACY_JAMMY_HEADERS_HOST
  "Also put the glibc-2.35 (jammy) header set in front of the HOST compilers. OFF: measured to break plain C++ TUs (stdlib.h __COLD) while the CUDA-only form already fixes the cospi/sinpi/rsqrt clash." OFF)

if(NOT NINFER_SM70_HOST_FLAGS_APPLIED AND NOT NINFER_LEGACY_HOST_FLAGS_APPLIED)
  set(NINFER_LEGACY_HOST_FLAGS_APPLIED 1)
  # Always: this toolkit warns that offline compilation for pre-75 targets will be
  # removed in a future release, on every TU, and the warning is not actionable here.
  set(CMAKE_CUDA_FLAGS_INIT
      "${CMAKE_CUDA_FLAGS_INIT} -Xcompiler=-isystem,${NINFER_JAMMY_HEADERS},-isystem,${NINFER_JAMMY_HEADERS}/x86_64-linux-gnu -Wno-deprecated-gpu-targets")
  if(NINFER_LEGACY_JAMMY_HEADERS_HOST)
    set(CMAKE_CXX_FLAGS_INIT  "${CMAKE_CXX_FLAGS_INIT} -isystem ${NINFER_JAMMY_HEADERS} -isystem ${NINFER_JAMMY_HEADERS}/x86_64-linux-gnu")
    set(CMAKE_C_FLAGS_INIT    "${CMAKE_C_FLAGS_INIT} -isystem ${NINFER_JAMMY_HEADERS} -isystem ${NINFER_JAMMY_HEADERS}/x86_64-linux-gnu")
  endif()
endif()

# --- what the rung is, said once, as a build fact ---------------------------
# NINFER_LEGACY_TIER=<rung> is the general name. NINFER_SM70_TIER is kept for
# the 70 rung alone because the sm_70 file defined it and notes written against
# that rung cite it. NEITHER IS A CAPABILITY CLAIM: a legacy rung has no tensor
# core (sm_5x/sm_6x) or only Volta's fp16 mma (sm_70), and which kernels a
# legacy DIST may actually run is decided at run time by the route selector
# (src/core/kernel_route.h) from the device, never from this macro.
add_compile_definitions(NINFER_LEGACY_TIER=${CMAKE_CUDA_ARCHITECTURES})
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  add_compile_definitions(NINFER_SM70_TIER=1)
endif()
