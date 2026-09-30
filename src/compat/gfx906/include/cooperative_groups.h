// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/compat/gfx906/include/cooperative_groups.h
// sha256(src) : 08e64ba1e546fefdc78ef25ea23b87ebbeb21f8536e234c8d351a5ac5881d93c
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : CUDA->HIP shim header. It SHADOWS a real CUDA header of the same name the moment src/compat/gfx906/include is put ahead of the CUDA include dir -- which is exactly what the fork does, include_directories(BEFORE ...) at src/CMakeLists.txt:5, tests/CMakeLists.txt:2 and bench/CMakeLists.txt:2. Not adding that line is what keeps it inert here. See docs/gfx906/PORT-AUDIT.md.
#pragma once
#include "core/hip_compat.h"

#include <hip/hip_cooperative_groups.h>
