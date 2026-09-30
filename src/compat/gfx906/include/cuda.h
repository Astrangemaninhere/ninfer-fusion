// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/compat/gfx906/include/cuda.h
// sha256(src) : a22c429c7f668a4c9f86d86443915a32aa13e60b78b1a3a13ac24f6a66bb3e84
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : CUDA->HIP shim header. It SHADOWS a real CUDA header of the same name the moment src/compat/gfx906/include is put ahead of the CUDA include dir -- which is exactly what the fork does, include_directories(BEFORE ...) at src/CMakeLists.txt:5, tests/CMakeLists.txt:2 and bench/CMakeLists.txt:2. Not adding that line is what keeps it inert here. See docs/gfx906/PORT-AUDIT.md.
#pragma once
// gfx906 port: the CUDA driver API is used only by the excluded NVFP4 TMA
// path. Nothing is mapped; retained code must not depend on it.
#include "core/hip_compat.h"
