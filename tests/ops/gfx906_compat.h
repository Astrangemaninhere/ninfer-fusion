// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : tests/ops/gfx906_compat.h
// sha256(src) : 4c97e7af84df4981c57d07884c082f0c806f7fb9a2abb1c50e08ea684ea1568c
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : Test-only format filter. Without NINFER_GFX906_COMPAT its kCompat constant is false and every suite that includes it is byte-for-byte unchanged. No kernel, dispatch or plan source is involved.
#pragma once

// gfx906 port (TP2 slice 3b): the split-op suites open with a registry sweep over EVERY format
// the donor registers, and their parity phases run NVFP4 / FP8 cases next to the groupwise ones.
// This port does not build NVFP4 or FP8 execution (gfx906_stubs.cpp throws "not supported on
// gfx906" by design), and ops::linear's bf16 tp2 shard has no gfx906-reachable launcher
// (bf16_dispatch.cpp rejects it: decode/small-T are exact-geometry, the MMA GEMM traps). Without
// a filter every split suite aborts in its registry phase within a second and the groupwise
// (Q4/Q5/Q6/W8) parity phase -- the one the shard routes on this card actually serve -- never runs.
//
// Under NINFER_GFX906_COMPAT `skip()` reports those formats as "skipped on gfx906: <qtype>" and
// the caller continues; on every other build it is false and the suites are unchanged. Test-only:
// no kernel, dispatch or plan source is involved.

#include "core/tensor.h"

#include <iostream>

namespace ninfer::test::gfx906 {

#if defined(NINFER_GFX906_COMPAT)
inline constexpr bool kCompat = true;
#else
inline constexpr bool kCompat = false;
#endif

inline const char* qtype_name(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return "Q4G64_F16S";
    case QType::Q5G64_F16S: return "Q5G64_F16S";
    case QType::Q6G64_F16S: return "Q6G64_F16S";
    case QType::W8G32_F16S: return "W8G32_F16S";
    case QType::BF16_CTRL: return "BF16_CTRL";
    case QType::FP32_CTRL: return "FP32_CTRL";
    case QType::I32_CTRL: return "I32_CTRL";
    case QType::NVFP4: return "NVFP4";
    case QType::FP8_E4M3FN_ROW_BF16S: return "FP8_E4M3FN_ROW_BF16S";
    }
    return "?";
}

// True when this build cannot execute `qtype` through a tp2 shard route. `bf16_linear_shard`
// says the route in question reaches ops::linear's bf16 tp2 shard (rejected on gfx906); the
// bf16 gdn_gating_proj shard launchers ARE built on gfx906, so callers on that route pass false.
inline bool unsupported(QType qtype, bool bf16_linear_shard = false) {
    if (!kCompat) { return false; }
    return qtype == QType::NVFP4 || qtype == QType::FP8_E4M3FN_ROW_BF16S ||
           (bf16_linear_shard && qtype == QType::BF16_CTRL);
}

// Logs and returns true when the caller should skip `what` for `qtype` on this build.
inline bool skip(QType qtype, const char* what, bool bf16_linear_shard = false) {
    if (!unsupported(qtype, bf16_linear_shard)) { return false; }
    std::cout << "skipped on gfx906: " << qtype_name(qtype) << " (" << what << ")\n";
    return true;
}

} // namespace ninfer::test::gfx906
