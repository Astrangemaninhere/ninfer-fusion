# cuda128_sm70_toolchain.cmake
# ---------------------------------------------------------------------------
# Host+device toolchain for the sm_70 (V100 / Volta) rung of ninfer-fusion.
#
# WHY THIS FILE EXISTS (two separate facts, both measured, see
# /home/user/scratch/PATCHSET/CUDA128/REPORT.md):
#
#  1. CUDA 13.x HAS NO compute_70  -> the sm_70 rung cannot be built by the
#     default toolkit at all; it needs the CUDA 12.8 toolkit in its own dir.
#
#  2. CUDA 12.8's nvcc refuses this distro's HOST image, in two independent
#     places, and neither of them is about sm_70:
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
# Usage (see REPORT.md for the full recipe):
#   cmake -S . -B build-sm70 -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=tools/archkit/cuda128_sm70_toolchain.cmake \
#         -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
# ---------------------------------------------------------------------------

set(NINFER_CUDA128_ROOT "/mnt/g/cuda12/tk"          CACHE PATH   "CUDA 12.8 toolkit root (the only toolkit here with compute_70)")
set(NINFER_HOST_GCC13     "/usr/bin/g++-13"         CACHE FILEPATH "host C++ for nvcc -ccbin; must be <= gcc 14")
set(NINFER_HOST_GCC13_C   "/usr/bin/gcc-13"         CACHE FILEPATH "host C for nvcc -ccbin")
set(NINFER_JAMMY_HEADERS  "/mnt/g/cuda12/jammy-sysroot/usr/include" CACHE PATH "glibc 2.35 header root that matches CUDA 12.8")

# --- device side -----------------------------------------------------------
set(CMAKE_CUDA_COMPILER      "${NINFER_CUDA128_ROOT}/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "${NINFER_HOST_GCC13}")
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_STANDARD 17)

# --- host side -------------------------------------------------------------
# Both compilers are pinned so that nvcc's host passes and the plain C++ TUs
# disagree about nothing; a mismatch here is how "it configured but would not
# link" happens.
set(CMAKE_C_COMPILER   "${NINFER_HOST_GCC13_C}")
set(CMAKE_CXX_COMPILER "${NINFER_HOST_GCC13}")

# --- the glibc 2.35 header redirection -------------------------------------
# -isystem (NOT -idirafter): -idirafter is searched AFTER the standard system
# dirs, so host mathcalls.h still wins and the 6 errors stay.  Measured.
# NOTE: this must stay ONE literal token, with commas and no ';'. A CMake LIST
# here expands to 'a;b', nvcc hands the ';' to sh, and the compiler-id probe then
# dies with `g++-13: fatal error: no input files` + `sh: 1: -isystem: not found`.
# Measured; do not "tidy" it back into a list.
# Guarded because CMake may read this toolchain file more than once, and an
# unguarded self-append then emits every flag twice in every compile command
# (harmless but it hides the real flags from the next reader). Measured.
if(NOT NINFER_SM70_HOST_FLAGS_APPLIED)
  set(NINFER_SM70_HOST_FLAGS_APPLIED 1)
  set(CMAKE_CUDA_FLAGS_INIT
      "${CMAKE_CUDA_FLAGS_INIT} -Xcompiler=-isystem,${NINFER_JAMMY_HEADERS},-isystem,${NINFER_JAMMY_HEADERS}/x86_64-linux-gnu -Wno-deprecated-gpu-targets")
  set(CMAKE_CXX_FLAGS_INIT  "${CMAKE_CXX_FLAGS_INIT} -isystem ${NINFER_JAMMY_HEADERS} -isystem ${NINFER_JAMMY_HEADERS}/x86_64-linux-gnu")
  set(CMAKE_C_FLAGS_INIT    "${CMAKE_C_FLAGS_INIT} -isystem ${NINFER_JAMMY_HEADERS} -isystem ${NINFER_JAMMY_HEADERS}/x86_64-linux-gnu")
endif()

# The rung is Volta: no bf16 MMA, no cp.async, no ldmatrix, no m16n8k16.
# Say it once here so a TU failing for that reason is obviously NOT a
# toolchain failure.
add_compile_definitions(NINFER_SM70_TIER=1)
