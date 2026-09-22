#pragma once

// Cut A: the TIER-FREE half of the small-T decode split policy.
//
// Everything here is host-side integer arithmetic over GqaExecutionEnvelope. It lives in its
// own header, separate from gqa_attention_decode_partial.cuh, because the routing TU
// (gqa_attention_decode.cu: gqa_attention_uses_small_t / gqa_attention_split_capacity) needs
// ONLY these entities -- while partial.cuh also includes the five per-tier kernel headers
// (decode_{bf16,fp8,iso3,i8,nvfp4}.cuh) and holds every tier's launch template in one
// anonymous namespace. The dispatcher's own body is two functions, so pulling partial.cuh in
// bought it the whole tier cascade for nothing.
//
// Moved verbatim from gqa_attention_decode_partial.cuh (no wording changes):
//   gqa_small_t_split_upper_bound / _reference / _count / _launch_capacity
// gqa_small_t_split_units<Geometry>() is NOT here: it stays where it belongs, in
// ops/kernel/gqa_attention_decode.cuh, and is __host__ __device__.
//
// These four used to sit in gqa_attention_decode_partial.cuh's anonymous namespace (internal
// linkage, one private copy per TU). Here they have external linkage on purpose: three of
// them are templates (ODR-safe) and the non-template gqa_small_t_split_reference is `inline`.
//
// Dependency budget (deliberate, keep it): ops/launcher/gqa_attention.h (the envelope and the
// public Tensor types it names), ops/common/math.h (div_up), core/dtype.h (DType), and
// ops/kernel/gqa_attention_decode.cuh (the Geometry traits plus gqa_small_t_split_units, which
// gqa_small_t_launch_capacity calls). NO per-tier kernel header, no __global__, no
// GqaSmallTInvocation. Pulling the kernel header in is safe and not a cycle: it includes only
// ops/common/{math,mma,warp}.cuh and ops/kernel/{gqa_attention_geometry,paged_kv_address}.cuh,
// never a launcher header, so nothing here can reach back to this file or to partial.cuh.
//
// INCLUDE PLACEMENT IS LOAD-BEARING -- include this header at FILE SCOPE only. Its body
// reopens `namespace ninfer::ops::detail`, so pulling it in from inside another namespace
// declares the entities in that nested namespace and every caller then sees them as
// undefined. The first cut of this patch included it from inside
// gqa_attention_decode_partial.cuh's anonymous namespace and failed to build for exactly that
// reason: gqa_attention_decode_smallt.cu: identifier "gqa_small_t_split_reference" undefined.

#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"                    // div_up
#include "core/dtype.h"                         // DType
#include "ops/kernel/gqa_attention_decode.cuh"  // gqa_small_t_split_units<Geometry>

#include <cstdint>

namespace ninfer::ops::detail {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

// The split reference of one launch: the pinned graph constant, or -- for callers that
// do not pin one yet -- the envelope itself (legacy behaviour).
//
// This is the ONLY reader of GqaExecutionEnvelope::split_reference_keys, and it is reached only
// from the split-KV small-T launches (gqa_attention_decode_smallt.cu:60-62 gives its value to both
// the partial kernels and the reducer). The prompt route does not call it and cannot: its launcher
// has no GqaExecutionEnvelope parameter. So the fallback below is the whole contract of the field
// -- pin it, or the partition follows the live window -- and a pinned caller must also be routed
// into this family (ops/launcher/gqa_attention_route_contract.h).
inline std::int32_t gqa_small_t_split_reference(GqaExecutionEnvelope envelope) {
    return static_cast<std::int32_t>(envelope.split_reference_keys != 0
                                         ? envelope.split_reference_keys
                                         : envelope.max_visible_keys);
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if ((kv_dtype == DType::I8 || kv_dtype == DType::E8Kv) && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if ((kv_dtype == DType::I8 || kv_dtype == DType::E8Kv) && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if ((kv_dtype == DType::I8 || kv_dtype == DType::E8Kv) && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    // NVFP4 first revision: coarser splits than INT8 to cut redundant Q
    // quantization and split-reduction overhead. Long contexts still scale.
    if (kv_dtype == DType::NVFP4) {
        // NINFER_GQA_SINGLE_WAVE=1 widens the short-window target to 128 so the split COUNT
        // matches the one-wave split_units (32 splits at a 4096 window instead of 64: 128 CTAs
        // instead of 256, half of which would exit immediately). Off by default -- 64 is the
        // value every baseline in this tree was taken under. This branch is NVFP4-only, so it
        // is inert for the other tiers, but it is NOT inert across geometries: see the coverage
        // table on kGqaDecodeWaveSplitsPerHead.
        //
        // gqa_single_wave_env() is defined unconditionally in ops/kernel/gqa_attention_decode.cuh
        // precisely so this header can call it from gqa_small_t_split_count (a plain host
        // function) in BOTH the host and the device pass of this TU. Do not move this call into
        // a `__host__ __device__` body -- that is where the guard is needed.
        const std::int32_t short_window_target =
            (gqa_single_wave_env() ? 128 : 64) / Geometry::DecodeSplitScale;
        const std::int32_t target =
            window > 16390 ? 480 / Geometry::DecodeSplitScale
                           : (window > 4096 ? 256 / Geometry::DecodeSplitScale
                                            : short_window_target);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        std::int32_t splits         = div_up(window, target);
        splits                      = splits > kMin ? splits : kMin;
        return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
    }
    // BF16, FP8_E4M3FN, and ISO3 share the generic split policy.
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t gqa_small_t_launch_capacity(GqaExecutionEnvelope envelope, std::int32_t tokens,
                                         DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            gqa_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    // Fixed split grid: div_up(max_visible_keys, split_units) splits already cover the
    // entire envelope and the device-side active count is exactly that, so the launched
    // grid is never smaller than div_up(window, split_units) for any window inside it.
    const std::int32_t split_units =
        gqa_small_t_split_units<Geometry>(gqa_small_t_split_reference(envelope));
    const std::int32_t fixed_splits =
        div_up(static_cast<std::int32_t>(envelope.max_visible_keys), split_units);
    capacity = capacity > fixed_splits ? capacity : fixed_splits;
    return capacity;
}
} // namespace ninfer::ops::detail
