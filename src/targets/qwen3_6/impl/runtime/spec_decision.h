#pragma once

// Per-request speculative backend decision policy (host-only, no CUDA deps).
//
// Measured on the 32 GiB RTX 5090 (Qwen3.8-27B NVFP4 artifact, 256-token
// greedy decodes, Chinese wiki prompts; /tmp/spec_threshold_results.txt):
//   prompt tok |  MTP d3 | DFlash2 d7
//        2,047 |   58.4  |   75.3  (+29%)
//        4,066 |   58.3  |   71.2  (+22%)
//        7,924 |   58.8  |   71.9  (+22%)
//       11,585 |   58.1  |   60.4  (+4%)
//       15,226 |   57.6  |   59.1  (+2.6%)
// MTP is flat to 21k+; the curves cross near 17-20k. DSpark is dominated at
// every point (65.8 at 2k, 35.3 at 15k) and excluded from selection.
//
// Policy (user decision 2026-08-30):
//   * admission: projected context <= kSpecDemoteTokens -> DFlash2, else MTP
//   * mid-flight: while DFlash2 is active, demote the engine to MTP once ALL
//     active requests' frontiers pass the threshold, or on VRAM pressure.
//     Demotion is low-frequency (at most once per growth crossing) and uses
//     the existing spec-degrade resume machinery.
//   * promotion back to DFlash2 happens only when the engine is fully idle.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::uint32_t kSpecDemoteTokens = 20480;

enum class SpecDecision : std::uint8_t {
    Mtp,
    DFlash2,
};

struct SpecAdmissionInputs {
    std::uint32_t prompt_tokens = 0;
    // Generation budget if the caller supplied one; 0 = unknown. When known,
    // a projection that is guaranteed to cross the threshold starts on MTP
    // directly instead of demoting mid-flight.
    std::uint32_t max_new_tokens = 0;
    bool dflash2_available       = false;
    std::size_t free_device_bytes      = 0;
    // Projected device bytes the DFlash2 context caches need for this request
    // (BF16 full-context draft caches). 0 = unknown / not tracked yet.
    std::size_t dflash2_context_bytes = 0;
};

[[nodiscard]] inline SpecDecision choose_spec_backend(const SpecAdmissionInputs& in) noexcept {
    if (!in.dflash2_available) { return SpecDecision::Mtp; }
    const std::uint64_t projection = static_cast<std::uint64_t>(in.prompt_tokens) +
                                     (in.max_new_tokens != 0 ? in.max_new_tokens : 0);
    if (projection > kSpecDemoteTokens) { return SpecDecision::Mtp; }
    if (in.dflash2_context_bytes != 0 && in.free_device_bytes != 0 &&
        in.dflash2_context_bytes > in.free_device_bytes) {
        return SpecDecision::Mtp;
    }
    return SpecDecision::DFlash2;
}

// Engine-wide mid-flight demotion check (called at decode round boundaries;
// cheap by design). Demote when every active request has crossed the
// threshold, or when the DFlash2 footprint no longer fits free VRAM.
[[nodiscard]] inline bool should_demote_dflash2(
    const std::vector<std::uint32_t>& active_frontiers, std::size_t free_device_bytes,
    std::size_t dflash2_footprint_bytes) noexcept {
    if (active_frontiers.empty()) { return false; }
    bool all_beyond = true;
    for (const std::uint32_t frontier : active_frontiers) {
        if (frontier <= kSpecDemoteTokens) {
            all_beyond = false;
            break;
        }
    }
    if (all_beyond) { return true; }
    return dflash2_footprint_bytes != 0 && free_device_bytes != 0 &&
           dflash2_footprint_bytes > free_device_bytes;
}

} // namespace ninfer::targets::qwen3_6
