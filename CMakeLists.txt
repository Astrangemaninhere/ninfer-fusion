cmake_minimum_required(VERSION 3.28)

# Multi-GPU build matrix (sm_75 .. sm_121, see tools/archkit/_GPU_MATRIX.md).
# Defaults to the native Blackwell target. A semicolon list is accepted; each
# entry is "<compute capability>" with an optional suffix: 'a' (arch-specific,
# 120a), or 'f' (family -- one binary for a whole sm_<N> family, e.g. 120f covers
# sm_120 AND sm_121; CUDA 12.9+).
#
# THE ARCH NUMBER DOES NOT CARRY THE CAPABILITY. Two builds with the same "120"
# do not accept the same KV formats: on an RTX 5090 D a 120a build reports all
# six KV formats SUPPORTED while a 120 build refuses fp8/nvfp4. The reason is
# that nvcc defines __CUDA_ARCH_FEAT_SM120_ALL only for the arch-specific ('a')
# target, and the fp4/TMA (tcgen05) kernels exist only under that macro. So the
# gate below checks the two facts that ARE facts, and names the missing part
# when it fails: (1) this toolchain can still generate code for the requested
# capability, and (2) whether the requested target carries the arch-specific
# feature set the default profile relies on. The number-only floor is gone: a
# floor is a property of the TOOLKIT, so the toolkit is asked (see the gate just
# below).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()

# --- CUDA arch capability gate -------------------------------------------------
# Asked BEFORE project() on purpose. project() runs its own "can this compiler
# build a trivial CUDA file for these archs?" test, so an arch the toolchain has
# dropped dies THERE first -- with a 30-line TryCompile dump and CMake's generic
# "is not able to compile a simple test program" instead of one sentence naming the
# arch. Locating nvcc here (the same places CMake looks: the cache, the CUDACXX
# environment variable, then PATH) lets the compiler's own answer come out first.
set(_ninfer_nvcc "")
if(DEFINED CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
  set(_ninfer_nvcc "${CMAKE_CUDA_COMPILER}")
elseif(DEFINED ENV{CUDACXX} AND EXISTS "$ENV{CUDACXX}")
  set(_ninfer_nvcc "$ENV{CUDACXX}")
else()
  find_program(_ninfer_nvcc NAMES nvcc HINTS /usr/local/cuda/bin ENV PATH)
endif()

set(_ninfer_supported_arches "")
set(_ninfer_arch_probe_rc "n/a")
set(_ninfer_arch_probe_error "the compiler was not found")
set(_ninfer_cuda_version_string "unknown version")
if(_ninfer_nvcc)
  execute_process(
    COMMAND "${_ninfer_nvcc}" --list-gpu-arch
    OUTPUT_VARIABLE _ninfer_arch_probe
    ERROR_VARIABLE _ninfer_arch_probe_error
    RESULT_VARIABLE _ninfer_arch_probe_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_ninfer_arch_probe_rc EQUAL 0)
    string(REGEX MATCHALL "compute_[0-9]+" _ninfer_supported_arches "${_ninfer_arch_probe}")
  endif()
  execute_process(COMMAND "${_ninfer_nvcc}" --version
                  OUTPUT_VARIABLE _ninfer_nvcc_version_output
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "release [0-9.]+" _ninfer_cuda_version_string
         "${_ninfer_nvcc_version_output}")
endif()

foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  # Syntax only -- no arch NUMBER is compared anywhere in this file.
  string(REGEX MATCH "^([0-9]+)([0-9])(a|f)?$" _arch_match "${_arch}")
  if(NOT _arch_match)
    message(FATAL_ERROR
      "NInfer: bad CUDA arch '${_arch}'. Expected the CMake spelling, which is the "
      "number plus an optional feature suffix: 75, 86, 89, 120a (arch-specific) or "
      "120f (family, one binary for sm_120 and sm_121).")
  endif()
  string(REGEX REPLACE "^([0-9]+)([0-9])(a|f)?$" "\\1\\2" _arch_num "${_arch}")
  string(REGEX MATCH "a$" _arch_is_feature "${_arch}")
  # The 'f' suffix is a THIRD thing, not a spelling of 'a'. nvcc defines
  # __CUDA_ARCH_FAMILY_SPECIFIC__ for it and does NOT define
  # __CUDA_ARCH_FEAT_SM<N>_ALL, so every guard in the tree that keys on the _ALL macro
  # alone reports a family build as "the kernels are not in this build". Detected
  # separately so the warning below cannot tell a family build it has no fp4 kernels.
  string(REGEX MATCH "f$" _arch_is_family "${_arch}")
  # Family targets were introduced in CUDA 12.9. A toolkit that lists compute_<N> but
  # predates them accepts the LIST check above and then dies inside project()'s TryCompile
  # with nvcc's own "Unsupported gpu architecture" -- a worse error, for a reason the caller
  # can act on. So when the toolkit answered, ask it this too.
  if(_arch_is_family AND _ninfer_supported_arches)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _ninfer_release_num "${_ninfer_cuda_version_string}")
    if(_ninfer_release_num AND _ninfer_release_num VERSION_LESS 12.9)
      message(FATAL_ERROR
        "NInfer: arch ${_arch} is a family target, and family targets exist only from "
        "CUDA 12.9; this toolkit is ${_ninfer_cuda_version_string}. Use a 12.9+ (or 13.x) "
        "toolkit, or ask for ${_arch_num}a / ${_arch_num} instead, which this toolkit does "
        "list.")
    endif()
  endif()
  if(_ninfer_supported_arches)
    # The compiler answered. This is the check, and it contains no hardcoded floor.
    if(NOT "compute_${_arch_num}" IN_LIST _ninfer_supported_arches)
      message(FATAL_ERROR
        "NInfer: this CUDA toolkit (${_ninfer_cuda_version_string}) cannot generate code for "
        "arch ${_arch}: compute_${_arch_num} is not in its own `nvcc --list-gpu-arch`. "
        "Requested: ${CMAKE_CUDA_ARCHITECTURES}. CUDA 13 removed offline code generation "
        "for Volta (sm_70) and older. Each GPU tier ships its own dist, so build this tier "
        "with a toolkit that still lists it (CUDA 12.x for sm_70) in its OWN build dir -- "
        "see tools/archkit/build_arch.sh, which carries the per-tier toolkit table -- "
        "instead of adding ${_arch} to a CUDA-13 fat list.")
    endif()
  elseif(_arch_num LESS 70 AND NOT CMAKE_TOOLCHAIN_FILE)
    # Fallback for a compiler that will not answer: the historical floor, kept only
    # so a broken probe still stops the build -- and it says that it is the
    # fallback rather than an independent judgement.
    message(FATAL_ERROR
      "NInfer: arch ${_arch} is below the floor this project has ever measured "
      "(compute capability 70, Volta). The `nvcc --list-gpu-arch` probe did not answer "
      "(rc=${_ninfer_arch_probe_rc}: ${_ninfer_arch_probe_error}), so a floor is applied "
      "instead of asking this compiler. NOTE the floor is 70, not 75: sm_70 is in scope, "
      "it is simply not buildable by CUDA 13 -- see tools/archkit/build_arch.sh for which "
      "toolkit builds which tier. If this IS a pre-75 build, name the toolkit that can "
      "still codegen it: -DCMAKE_TOOLCHAIN_FILE=tools/archkit/cuda128_legacy_toolchain.cmake "
      "(which supplies the compiler this probe could not find), or set CUDACXX / "
      "-DCMAKE_CUDA_COMPILER to a CUDA 12.x nvcc so this probe can answer for itself.")
  elseif(_arch_num LESS 70)
    # CMAKE_TOOLCHAIN_FILE is a cache entry, so it IS visible here even though the
    # file itself is not read until project(). A named toolchain file means the
    # compiler is the caller's decision and this probe's silence is not evidence
    # about the arch -- so the floor is NOT applied, and the compiler is asked for
    # real by project()'s TryCompile (nvcc's own "Unsupported gpu architecture"
    # is then the error, for an arch that genuinely cannot be built).
    message(STATUS
      "NInfer: arch ${_arch} is below the historical floor (70), and this probe could not "
      "answer (rc=${_ninfer_arch_probe_rc}: ${_ninfer_arch_probe_error}) -- but "
      "CMAKE_TOOLCHAIN_FILE is set (${CMAKE_TOOLCHAIN_FILE}), i.e. the caller named the "
      "toolkit that supplies the compiler, so the floor is not applied. The compiler's own "
      "answer is what decides this build now.")
  endif()
  if(_arch_is_feature)
    # What nvcc defines for this target, and the ONLY thing that distinguishes the 'a'
    # spelling from the same arch number WITHOUT any suffix. (There is a third spelling,
    # 'f', handled in the elseif below; it is distinguished by a different macro.) Named
    # so the next reader can grep for it (tools/archkit/_GPU_MATRIX.md says the same
    # thing in prose).
    set(_ninfer_feature_macro "__CUDA_ARCH_FEAT_SM${_arch_num}_ALL")
    if(NOT _ninfer_supported_arches AND _arch_num GREATER_EQUAL 100)
      message(STATUS
        "NInfer: arch ${_arch} requests the arch-specific feature set "
        "(__CUDA_ARCH_FEAT_SM${_arch_num}_ALL); that could not be confirmed against the "
        "compiler's arch list (probe rc=${_ninfer_arch_probe_rc}).")
    endif()
  elseif(_arch_is_family)
    # A family build shares ONE binary across a whole sm_<N> family (120f covers sm_120
    # and sm_121). It is not "an 'a' build without the suffix": nvcc defines
    # __CUDA_ARCH_FAMILY_SPECIFIC__ and NOT __CUDA_ARCH_FEAT_SM${_arch_num}_ALL, so
    # capability guards that only know the _ALL macros will answer "not in this build"
    # for kernels ptxas demonstrably assembles for that target. Say which macro to key on
    # rather than leaving that to be rediscovered.
    message(STATUS
      "NInfer: arch ${_arch} is a CUDA family target: nvcc defines "
      "__CUDA_ARCH_FAMILY_SPECIFIC__ (and NOT __CUDA_ARCH_FEAT_SM${_arch_num}_ALL) for it, "
      "and one binary serves the whole sm_${_arch_num} family. Capability guards must "
      "accept the family macro as well as the _ALL ones -- see the guard block in "
      "src/core/device_probe.cu and the feature gates in src/ops/common/mma.cuh.")
  elseif(_arch_num GREATER_EQUAL 100)
    message(WARNING
      "NInfer: arch ${_arch} has no 'a' suffix, so nvcc will NOT define "
      "__CUDA_ARCH_FEAT_SM${_arch_num}_ALL for it. The arch-specific fp4/TMA (tcgen05) "
      "kernels are compiled OUT of that target, and the matching KV formats are refused "
      "at run time. Use ${_arch}a if you want them; keep ${_arch} only for the "
      "groupwise-int profile. Same arch number, different capability -- "
      "tools/archkit/_GPU_MATRIX.md.")
  endif()
endforeach()
unset(_arch)
unset(_arch_num)
unset(_arch_match)
unset(_arch_is_feature)
unset(_arch_is_family)
unset(_ninfer_release_num)
unset(_ninfer_feature_macro)
unset(_ninfer_arch_probe)
unset(_ninfer_arch_probe_error)
unset(_ninfer_arch_probe_rc)
unset(_ninfer_supported_arches)
unset(_ninfer_nvcc_version_output)

# ---------------------------------------------------------------------------------------------
# ccache launcher (2026-09-19, gatebypass)
#
# WHY THIS IS IN THE SOURCE AND NOT THE BUILD CACHE
#   build/CMakeCache.txt:40 pins `CMAKE_CUDA_COMPILER_LAUNCHER:UNINITIALIZED=ccache`. Type
#   UNINITIALIZED means it came from a `-D` (or the same-named env var) at configure time --
#   **no file in this repository ever set it** (root CMakeLists.txt had zero ccache/launcher
#   matches). A cache variable survives every reconfigure, and sh2/_mg3_reconfigure.sh:30 runs
#   `cmake -S . -B build` with no launcher argument, so **it re-burned the pin on every run**.
#   A fix that a reconfigure can silently undo is not a fix => the pin must be resolved here.
#
# WHY IT IS DELIBERATELY "LOUD WHEN ABSENT"
#   With the launcher set to a ccache that does not exist, every CUDA compile dies with
#   `/bin/sh: 1: ccache: not found` (exit 127) and the tree cannot build at all. That is a LOUD
#   failure and it is the behaviour to preserve. Two shims have been installed to silence it --
#   a /usr/local/bin/ccache passthrough (deleted; its real cost was turning a loud exit 127 into
#   a SILENT non-caching build) and, tonight, a script-local passthrough in
#   dl/autodraft/_build.sh:14-24. **This block must never be "fixed" by installing a third.**
#   When ccache is absent we CLEAR the stale cache entry, so the compile runs with no launcher
#   prefix and no cache -- never with a launcher that does not resolve.
#
# WHY unset(... CACHE) RATHER THAN set(... FORCE)
#   unset removes only the stale entry, so an operator who deliberately passes
#   -DCMAKE_CUDA_COMPILER_LAUNCHER=<something> on a later configure still wins. FORCE would be
#   stricter but would silently discard that choice. The requirement here is only that the
#   *stale, unresolvable* value cannot survive a reconfigure.
# ---------------------------------------------------------------------------------------------
find_program(NINFER_CCACHE_EXECUTABLE NAMES ccache)
# ⚠️ find_program() CACHES its result. Without the re-check below, the sequence
#   "configure while ccache exists -> ccache is later removed -> reconfigure"
# leaves the launcher pinned at a path that no longer exists -- i.e. exit 127 comes back, from a
# NEW cause, and the cache is again the thing that silently holds it. Measured in
# dl/gatebypass/dryrun/harness2.sh T8c. So: if the cached path is gone, drop it and search again.
if(NINFER_CCACHE_EXECUTABLE AND NOT EXISTS "${NINFER_CCACHE_EXECUTABLE}")
  message(STATUS "ninfer: cached ccache '${NINFER_CCACHE_EXECUTABLE}' no longer exists; re-searching")
  unset(NINFER_CCACHE_EXECUTABLE CACHE)
  find_program(NINFER_CCACHE_EXECUTABLE NAMES ccache)
endif()
if(NINFER_CCACHE_EXECUTABLE)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${NINFER_CCACHE_EXECUTABLE}")
  set(CMAKE_CXX_COMPILER_LAUNCHER  "${NINFER_CCACHE_EXECUTABLE}")
  message(STATUS "ninfer: compiler launcher = ${NINFER_CCACHE_EXECUTABLE}")
else()
  if(CMAKE_CUDA_COMPILER_LAUNCHER OR CMAKE_CXX_COMPILER_LAUNCHER)
    message(WARNING
      "ninfer: no ccache found on PATH; CLEARING the cached compiler launcher "
      "'${CMAKE_CUDA_COMPILER_LAUNCHER}'. Leaving it would make every CUDA compile fail with "
      "exit 127 (/bin/sh: ccache: not found). Builds will run uncached and correct. "
      "To restore caching, install ccache and reconfigure -- do NOT install a passthrough shim.")
  endif()
  unset(CMAKE_CUDA_COMPILER_LAUNCHER CACHE)
  unset(CMAKE_CXX_COMPILER_LAUNCHER  CACHE)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
# THE TEST-ONLY ARCHITECTURE SIMULATOR KEY, and the FIRST of the three keys
# src/core/arch_sim.h requires. DEFAULT OFF, and the default is the product statement: a
# shipping binary must not be able to be talked into simulating a card it is not, by an
# environment variable alone. On, it defines NINFER_ARCH_SIM_ENABLED on the engine targets
# that include arch_sim.h (src/CMakeLists.txt), which is what makes the simulator a real
# build-time capability instead of a header nobody's TU includes.
#
# MEASURED 2026-09-18 (SIM-RUN): this option did not exist. `NINFER_ENABLE_ARCH_SIM`
# appeared in the tree only inside comments and one test's assertion text, so
# `cmake -DNINFER_ENABLE_ARCH_SIM=ON` set nothing, and `NINFER_ARCH_SIM_ENABLED` was defined
# on exactly one target (tests/CMakeLists.txt:169, the host test). The header's own
# three-key story was therefore true as prose and unexecutable as a recipe.
option(NINFER_ENABLE_ARCH_SIM
  "Build the TEST-ONLY architecture simulator (src/core/arch_sim.h) into the engine. OFF by default: a shipping binary cannot be talked into simulating a card it is not by an environment variable alone." OFF)

# THE TEST-ONLY VENDOR-AXIS OVERRIDE KEY (src/core/vendor_sim.h), and the FIRST of the three keys
# that header requires. DEFAULT OFF, and it is a SEPARATE OPTION FROM NINFER_ENABLE_ARCH_SIM ON
# PURPOSE: reusing the arch-sim key would mean a binary already audited for the V100 simulator
# silently acquired a second axis, which is a widening of a landed gate. Two axes, two keys.
#
# ⚠ AND THIS OPTION GATES THE TEST BINARY ONLY, WHICH IS A MEASUREMENT AND NOT A SAVING. The
# arch-sim option above gates an engine block in src/CMakeLists.txt because src/targets/registry.cpp
# and src/ops/qpn/qpn_arch_route.cpp include arch_sim.h. NOTHING in src/ includes vendor_sim.h,
# because the AMD apparatus in src/core/arch_caps.h has ZERO callers outside that header. A define
# on a TU that includes nothing changes nothing -- src/CMakeLists.txt measures exactly that case for
# the arch axis ("compiling src/targets/registry.cpp with and without the key produced
# BYTE-IDENTICAL objects ... because no engine TU included arch_sim.h at all"). So the vendor key
# is consumed by tests/CMakeLists.txt alone, and the day an engine consumer lands, an
# if(NINFER_ENABLE_VENDOR_SIM) block lands in src/CMakeLists.txt WITH it.
#
# The OFF default is still the product statement, and it is provable by inspection: with the option
# off, `strings <shipping binary> | grep -c NINFER_SIM_VENDOR` is 0, because only the test binary
# carries the key and the test binary is not shipped.
option(NINFER_ENABLE_VENDOR_SIM
  "Build the TEST-ONLY vendor-axis capability override (src/core/vendor_sim.h) into the TEST binary. OFF by default; it gates tests/CMakeLists.txt only, because no engine TU includes that header (measured)." OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Switch-coverage gate (w-fit8 T1) ----------------------------------------
#
# WHY THIS EXISTS, in one measured sentence: `w_fit6`'s P4 removed the `default:`
# arm from ProgramImplCore::memory_summary()'s `storage_of` lambda so a future
# DType enumerator would be REPORTED instead of silently resolving to
# KvCacheStorage::BFloat16 (the per-layer "inherit the global --kv-dtype"
# sentinel, i.e. a silent no-op reported as success) -- and on this tree that
# protection was INERT, because CMakeLists.txt opened no warning flag at all.
# w-fit8 measured it: with 12 DTypes and no flag the carrier TU exits 0 and prints
# nothing from -Wswitch; the only default-on diagnostic is -Wreturn-type.
#
# Three states. Only the third is a hard gate.
#   off    : no flag (historical). P4 is INERT here.
#   report : add the checks as warnings. Zero build risk on this tree (w-fit8
#            measured 326/326 host TUs still rc=0), and P4 now PRINTS.
#   error  : report + -Werror for exactly the checks in NINFER_WARN_ERROR_CHECKS.
#
# !! DO NOT name `switch-enum` in NINFER_WARN_ERROR_CHECKS on this tree. !!
# w-find5 measured -Wswitch-enum at 767 warnings over 19 files on this tree, and
# most of those switches name only part of their enumeration ON PURPOSE. Escalating
# that check turns 767 warnings into 767 build errors -- that is not a landing, it
# is a rewrite of 19 files. The check P4 needs is plain `-Wswitch` ("an arm is
# missing AND there is no default"), which is a strict subset of -Wswitch-enum and
# is what w-fit8 measured the carriers against.
#
# SCOPE, deliberately narrowed from w-find5's draft: CXX only, no nvcc. P4's lambda
# is a host header reached only through the four host variant.cpp carriers
# (verified: program_impl.h is reached solely via
# src/targets/qwen3_6/impl/runtime/instantiate.h), so a CUDA half would add an
# UNMEASURED surface for no coverage. Add it in a separate change, with .cu
# evidence.
set(NINFER_WARN_SHADOW_SWITCH "off" CACHE STRING
    "Switch-coverage gate: off | report | error")
set_property(CACHE NINFER_WARN_SHADOW_SWITCH PROPERTY STRINGS off report error)
set(NINFER_WARN_ERROR_CHECKS "switch" CACHE STRING
    "Checks -Werror applies to when NINFER_WARN_SHADOW_SWITCH=error "
    "(semicolon list; 'switch' is the intended value on this tree)")

# --- the warning surface this repository asks for ITSELF --------------------
# Prepared by w-warnsurface (dl/warnsurface/REPORT.md, 2026-09-18). MEASURED: these
# TUs compile with no warning flag of their own (build/CMakeCache.txt has
# CMAKE_CXX_FLAGS empty, and this file refuses -Wall by design), so the printf-family
# checks below were live ONLY because Ubuntu's GCC >= 13 injects
# `-Wformat -Wformat-security` (and >= 15 `-Wbidi-chars=any`) through its
# *distro_defaults spec. On any toolchain that does not carry that spec -- upstream
# GCC, clang, a container, a Windows toolchain, a vendor CUDA host -- the checks
# disappear with no diff to this repository, and they are warnings, never errors, so
# even where they fire the build still succeeds.
#
# Cost of making them explicit, MEASURED (three representative TUs, build's own
# command line, objects compared): apps/cli/main.cpp 0 -> 0, apps/cli/options.cpp
# 0 -> 0, the program_impl.h carrier src/targets/qwen3_6_27b/impl/variant.cpp 1 -> 1
# (the same pre-existing -Wreturn-type). Zero new diagnostics. -Wnonnull arrives with
# -Wformat; it was measured enabled only under this group.
#
# NOTE: -Wall does NOT suppress these -- Ubuntu's spec skips its own -Wformat
# injection once -Wall is present (measured). Adding them is therefore orthogonal to
# the -Wall question, and it is a configure-time change: it rewrites flags.make and
# re-stales every TU. Hold it for the next configure, together with the two other
# held CMake items (ccache find_program: gatebypass; ninfer_build_cuda_arches
# INTERFACE target: routeprobe) -- one configure, three items.
#
# NOT included, deliberately: the CUDA half (nvcc's host pass inherits the injection
# from the host gcc and would need -Xcompiler=-Wformat...). Unmeasured for cost, and
# this tree's own precedent for its -Wswitch gate is to land a CUDA half separately
# with .cu evidence. The C half (1 .c TU in the build) is likewise unmeasured.
#
# A CACHE toggle in the NINFER_WARN_SHADOW_SWITCH house style is possible if the
# fleet prefers it; the three lines below are the minimal form.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wformat>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wformat-security>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wbidi-chars=any>)

if(NOT NINFER_WARN_SHADOW_SWITCH STREQUAL "off")
  # -Wswitch is enabled by -Wall, but -Wall is NOT used here: on this tree it would
  # also open ~600 unrelated warnings, which is a different decision from this gate.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wswitch>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wswitch-enum>)
endif()
if(NINFER_WARN_SHADOW_SWITCH STREQUAL "error")
  foreach(_ninfer_chk IN LISTS NINFER_WARN_ERROR_CHECKS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror=${_ninfer_chk}>)
  endforeach()
  unset(_ninfer_chk)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()
# --- nvcc temporaries: off /tmp, and off RAM --------------------------------
#
# MEASURED 2026-09-18 (tmpcliff): /tmp on this box is tmpfs, so nvcc's
# cudafe/cicc/ptxas intermediates -- written as /tmp/tmpxft_<pid8hex>_<seq8hex>*
# -- are RAM pages taken from the whole build fleet. nvcc removes them only when
# a compile SUCCEEDS (leftovers=0 in all ten probed cases), so what accumulates
# is the orphans of KILLED builds, in STEPS: 52 such files measured at 1.697 GiB,
# ~95 % of everything in /tmp, and each dead fat unit costs ~750 MiB (one
# 751.7 MiB .ptx). A linear time-to-cap forecast is therefore the wrong model.
#
# --objdir-as-tempdir keeps those intermediates beside the OBJECT
# (<obj>.compute_NN.ptx, ...) instead of under TMPDIR, so a killed build spends
# the tree's ext4 disk instead of the 11 GiB RAM cap. MEASURED on CUDA 13.3 with
# --keep: 23 intermediates in the object's own directory, 0 tmpxft_* anywhere.
# Toolkit-portable, and that matters here: build-52 / build-61 / build-sm70 are
# built by CUDA 12.8 (/mnt/g/cuda12/tk) and the flag is present in BOTH toolkits
# (13.3 nvcc --help:145, 12.8 nvcc --help:122, both spelled `-objtemp`).
#
# Added via add_compile_options, NOT by appending to CMAKE_CUDA_FLAGS: a bare
# set() would shadow each build dir's CACHED CMAKE_CUDA_FLAGS for a single
# configure and not persist it, and build-52/61/sm70 carry
# `-Xcompiler=-isystem,/mnt/g/cuda12/jammy-sysroot/...` in that cache, which such
# a shadow would silently drop on the next reconfigure. The generator expression
# is cache-independent and is re-applied by every configure.
#
# This is the BELT. The braces are TMPDIR=/var/tmp/nvcc-tmp in ~/.profile.
# Neither removes bytes written -- only where they land.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>)

# The toolkit floor is a property of the TARGET LIST, not of the project.
# CUDA 13.x is needed for the arch-specific ('a') and family ('f') suffixes and
# for sm_121 and above. A purely numeric list at or below 120 needs none of
# those, and a CUDA 12.x toolkit is the ONLY family that still generates code for
# Volta (sm_70). Making the check conditional is what lets the sm_70 rung be
# built at all; leaving it unconditional rejected the only toolkit that has it.
#
# STATUS OF THE 12.x BRANCH: the GATE half of it is MEASURED. A CUDA 12.8 toolkit
# (/mnt/g/cuda12/tk, nvcc V12.8.61) answers `--list-gpu-arch` with compute_70 and without
# compute_121/103/110, and against it this gate accepts 70/72/75/120/120a and rejects
# 121/121a/110a. What is still NOT measured on this box is the COMPILE half: 12.8's own
# host_config.h rejects the system gcc 15 ("unsupported GNU version"), -allow-unsupported-
# compiler then dies on gcc-15 builtins (__is_pointer/__is_volatile/__array_rank), and
# -ccbin g++-13 dies on the glibc-vs-CUDA-12.8 mathcalls exception-specification clash
# (cospi/sinpi/rsqrt). So no target of this tree has been BUILT by 12.8 here -- only the
# gate has. A matching toolchain image (nvidia/cuda:12.8.0-devel-ubuntu22.04) is what
# closes that; the argv and criteria are in the MULTIARCH2 report.
set(_ninfer_needs_cuda13 OFF)
foreach(_ninfer_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "^([0-9]+)([0-9])(a|f)?$" "\\1\\2" _ninfer_floor_num "${_ninfer_arch}")
  if(_ninfer_floor_num GREATER_EQUAL 121)
    set(_ninfer_needs_cuda13 ON)
  endif()
  if(_ninfer_arch MATCHES "(a|f)$")
    set(_ninfer_needs_cuda13 ON)
  endif()
endforeach()
unset(_ninfer_arch)
unset(_ninfer_floor_num)
if(_ninfer_needs_cuda13 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer: arch list '${CMAKE_CUDA_ARCHITECTURES}' needs CUDA 13.1 or newer, but the "
    "CUDA compiler is ${CMAKE_CUDA_COMPILER_VERSION}. The arch-specific ('a') and family "
    "('f') suffixes and sm_121 exist only in CUDA 13.x. Use a 13.x toolkit, or request "
    "only plain numeric archs at or below 120.")
endif()
if(NOT _ninfer_needs_cuda13 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR
    "NInfer: a plain numeric arch list needs CUDA 12.0 or newer; found "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# The arch capability gate lives ABOVE project() (it needs only nvcc, not a
# configured project) so that a rejected arch is reported as one sentence naming
# the arch, instead of project()'s TryCompile failing first. Nothing arch-related
# is decided after this point.

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------

# GNUInstallDirs defines CMAKE_INSTALL_BINDIR / _DOCDIR / _DATADIR / _LIBDIR. It is included
# here, ahead of the subdirectories, because apps/CMakeLists.txt's install(TARGETS) rule uses
# CMAKE_INSTALL_BINDIR. Writing the literal "bin" there instead would have made DESTDIR and
# `cmake --install --prefix` behave differently across platforms for no gain.
include(GNUInstallDirs)

add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# --- Deployment: what `cmake --install` produces -------------------------------
#
# WHY HERE AND NOT IN A SUBDIRECTORY. The three build switches are declared in this file, so
# this is the only place where the guard that pairs a target with its install rule can be read
# next to the switch that creates the target. apps/CMakeLists.txt owns the executables' own
# rule; this block owns everything that is not a target, and therefore must run regardless of
# NINFER_BUILD_APPS / BUILD_TESTING / NINFER_BUILD_BENCHMARKS.
#
# MEASURED 2026-09-20 (dl/winstage2): before this block `install(` had ZERO matches over all 13
# tracked CMakeLists.txt, and the only matches for the word "install" in this file were three
# comments (:195, :198, :229). `cmake --install` therefore did not exist as a command here.
#
# WHAT IS DELIBERATELY NOT INSTALLED, so a later reader does not read the omission as an
# oversight:
#   * the engine static libraries -- no exported headers, so the archive is unusable alone;
#   * the Dockerfile -- it describes a container build, not an install tree;
#   * tests/, bench/, tools/ -- excluded from the default build by design (README.md:40);
#   * docs/ -- 30 of its 55 on-disk files are tracked by git, so installing the directory
#     would make two installs from the same revision differ. Narrowing that is a separate job.
install(FILES
    "${PROJECT_SOURCE_DIR}/README.md"
    "${PROJECT_SOURCE_DIR}/LICENSE"
    "${PROJECT_SOURCE_DIR}/CONTRIBUTING.md"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}"
  COMPONENT documentation)

# An install tree that describes itself. Generated at configure time into the build tree and
# installed from there, so it needs no source tree at install time and cannot drift from the
# settings that actually produced the binaries next to it.
set(_ninfer_manifest_path "${CMAKE_CURRENT_BINARY_DIR}/ninfer-install-manifest.txt")
file(WRITE "${_ninfer_manifest_path}"
"ninfer install manifest
"
"=======================
"
"Written by ${CMAKE_PROJECT_NAME}'s CMakeLists.txt at configure time and installed by
"
"`cmake --install`.
"
"
"
"source directory   : ${PROJECT_SOURCE_DIR}
"
"build directory    : ${PROJECT_BINARY_DIR}
"
"build type         : ${CMAKE_BUILD_TYPE}
"
"install prefix     : ${CMAKE_INSTALL_PREFIX}
"
"                     (overridable at install time: cmake --install <build-dir> --prefix <path>)
"
"install bindir     : ${CMAKE_INSTALL_BINDIR}
"
"install docdir     : ${CMAKE_INSTALL_DOCDIR}
"
"CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}
"
"C++ compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
"
"CUDA compiler      : ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}
"
"host system        : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}
"
"
"
"COMPONENTS
"
"  runtime       : ninfer, ninfer-serve, ninfer-perplexity  -> <prefix>/${CMAKE_INSTALL_BINDIR}
"
"  documentation : this file, README.md, LICENSE, CONTRIBUTING.md -> <prefix>/${CMAKE_INSTALL_DOCDIR}
")
install(FILES "${_ninfer_manifest_path}"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}"
  COMPONENT documentation)
