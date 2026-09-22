#pragma once

// ninfer::ops::detail - the GQA route decision, as a pure host function.
//
// This header exists so the DECISION is testable without a GPU and without linking the ops
// library: gqa_attention_resolve_route (src/ops/wrapper/gqa_attention.cpp) is a one-line
// forwarder onto gqa_attention_route_for below, so a host test that calls the function here
// calls the dispatcher's own decision rather than a copy of it.
//
// The constants are spelled again here instead of being moved out of gqa_attention.cpp's
// anonymous namespace (moving them would touch every use inside that TU); gqa_attention.cpp
// static_asserts equality against its own copies, so the two spellings cannot drift apart.

#include "ops/common/math.h"  // div_up
#include "ops/kernel/gqa_attention_geometry.cuh" // Gqa16x4Geometry
#include "ops/launcher/gqa_attention.h"          // GqaAttentionRoute, GqaExecutionEnvelope

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kGqaSmallTChunkTokens            = 6;
inline constexpr std::int32_t kGqaMaximumVerifyTokens          = 16;
inline constexpr std::uint32_t kGqaTwoChunkPromptVisibleKeys   = 512;
inline constexpr std::uint32_t kGqaThreeChunkPromptVisibleKeys = 1024;

// The route of one launch. Body moved verbatim out of gqa_attention.cpp (it was
// gqa_attention_resolve_route there) with ONE added clause, marked (*) below.
inline GqaAttentionRoute gqa_attention_route_for(std::int32_t q_heads, std::int32_t width,
                                                 std::int32_t batch_size,
                                                 GqaExecutionEnvelope envelope) {
    if (width >= 1 && width <= kGqaSmallTChunkTokens) { return GqaAttentionRoute::SmallT; }
    if (batch_size > 1) { return GqaAttentionRoute::ChunkedSmallT; }
    const std::uint32_t prompt_visible_keys =
        width <= 2 * kGqaSmallTChunkTokens ? kGqaTwoChunkPromptVisibleKeys
                                           : kGqaThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kGqaMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    // (*) A launch that PINNED its split reference is a decode/verify whose columns must reduce
    // exactly like the batch-1 decode of the same row (GqaExecutionEnvelope::split_reference_keys,
    // include/ninfer/ops/gqa_attention.h:18-25). The prompt route cannot carry that pin: its
    // launcher is gqa_attention_prompt_launch, which takes no GqaExecutionEnvelope at all
    // (src/ops/launcher/gqa_attention.h:69-72), and its body partitions keys into kGqaPrefillBc
    // (64) blocks with its own sequential online-softmax rescale
    // (src/ops/kernel/gqa_attention_prefill_bf16.cuh:140, :251, :263, :394-395, :453-456).
    // A row's fp32 reduction would therefore stop being a function of the row's own key prefix,
    // which is the whole point of the pin. The chunked small-T route DOES consume it -- every
    // launch it emits goes through gqa_attention_split_capacity(q_heads, tokens, dtype, envelope)
    // and so through gqa_small_t_split_units on gqa_small_t_split_reference -- so a pinned
    // multi-column verify goes there instead of to the prompt body.
    //
    // What this clause does NOT claim: it does not make the two routes numerically identical. The
    // chunked small-T route reproduces the batch-1 decode because the pinned grid makes a row's
    // split partition a function of (its own key prefix, split_units) alone -- the partial kernel
    // clips every split to the live window and the trailing splits it adds for a wider launch carry
    // the neutral partials (m = -inf, l = 0) the reducer skips. Asserting that end to end needs a
    // GPU; the host test pins the mechanism and the route contract.
    //
    // Gqa16x4Geometry (Spark-X2.5, 16 q-heads / 4 kv-heads) is covered by `q_heads != 16`: the
    // small-T launcher dispatches 16 q-heads at head_dim 256 to the Gqa35Geometry (16/2) instance
    // and has no kv-head signature to separate the two, so moving a 16-head call off the prompt
    // body here could hand a 16/4 cache to a 16/2 kernel. A pinned 16-head wide verify therefore
    // still reaches the prompt route. That residual is named in the report, not silently accepted.
    if (envelope.split_reference_keys != 0 && width <= kGqaMaximumVerifyTokens &&
        q_heads != Gqa16x4Geometry::QHeads) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    return GqaAttentionRoute::Prompt;
}

// Whether a route consumes GqaExecutionEnvelope::split_reference_keys. SmallT and ChunkedSmallT
// both reach gqa_small_t_split_reference; Prompt never sees the envelope. Kept next to the
// decision so the test and the report assert one predicate instead of three spellings of it.
[[nodiscard]] inline bool gqa_attention_route_consumes_split_pin(GqaAttentionRoute route) {
    return route != GqaAttentionRoute::Prompt;
}

// The chunked small-T decomposition of one launch, as the number of launches: consecutive runs of
// at most kGqaSmallTChunkTokens columns. This is what gqa_attention()'s launch_chunked_small_t
// applies (src/ops/wrapper/gqa_attention.cpp:413-425, `for (begin = 0; begin < q.ne[2];
// begin += kSmallTChunkTokens)`), restated here so the test can assert that every column of a
// pinned wide verify lands in a chunk that the pinned grid covers.
[[nodiscard]] inline std::int32_t gqa_small_t_chunk_count(std::int32_t width) {
    if (width <= 0) { return 0; }
    return div_up(width, kGqaSmallTChunkTokens);
}

} // namespace ninfer::ops::detail
