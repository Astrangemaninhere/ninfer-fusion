// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/ops/linear/gfx906/stage8_route.h
// sha256(src) : 8e6dbaf84d565dce0f8f2e7fe7cdc41bdc077c8df5ba18a1b9ed5b686725f9c8
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : Pure host C++ (its only include is <cstdlib>). This is the ONE file in this set that both compiles and runs on this box; its two env gates (NINFER_GFX906_STAGE8, NINFER_GFX906_PASS2) were exercised here rather than assumed -- see gfx906port REPORT control P1.
#pragma once

// Stage-8 route gate for the gfx906 wave64 tiled GEMM kernels.
//
// NINFER_GFX906_STAGE8=0 in the environment restores the stage-3 SIMT
// fallback routing (the A/B lever for the Tier-1/Tier-2 benchmarks); any
// other value, or an unset variable, keeps the tiled kernels on.

#include <cstdlib>

namespace ninfer::ops::detail {

inline bool gfx906_stage8_tiled_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GFX906_STAGE8");
        return !(value != nullptr && value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// Pass-2 route gate for the gfx906 register-resident T=1 GEMV kernels
// (q_gemv_gfx906.cuh). NINFER_GFX906_PASS2=0 restores the upstream
// q5_rowsplit_gemv route (the A/B lever); unset or any other value keeps the
// pass-2 kernels on. Separate from NINFER_GFX906_STAGE8 so the two retunes can
// be A/B'd independently.
inline bool gfx906_pass2_gemv_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GFX906_PASS2");
        return !(value != nullptr && value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// Pass-2e route gate for the small-T (2..5) variants of the pass-2 GEMV
// kernels (the MTP verify shapes). Requires NINFER_GFX906_PASS2 on;
// NINFER_GFX906_PASS2_SMALLT=0 restores the previous T>1 routes (the stage-8
// tiled GEMM / Materialized paths) for a separate A/B.
inline bool gfx906_pass2_smallt_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GFX906_PASS2_SMALLT");
        return gfx906_pass2_gemv_enabled() &&
               !(value != nullptr && value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

} // namespace ninfer::ops::detail
