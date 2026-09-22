#pragma once

#include "runtime/contract/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::runtime_support {

inline void include_rebuild_boundary(std::uint32_t& tail_begin, std::uint32_t boundary,
                                     std::uint32_t frontier) noexcept {
    if (boundary <= frontier) { tail_begin = std::max(tail_begin, boundary); }
}

// Advance the segmented rebuild accounting from `previous_frontier` to `frontier`.
//
// `work.chunks` is a running sum of prefill UNITS:
//     (units charged for [0, tail_begin)) + (units charged for [tail_begin, frontier))
// The tail is the part that gets re-prefilled when a checkpoint is dropped, so the
// second term has to be replaced (not grown) on every advance, which is why this
// subtracts the old tail's contribution and adds the new one.
//
// The unit is NOT a constant. bandwidth_governor_.prefill_chunk_for() installs a
// value in [128, prefill_chunk_capacity()] between two calls (engine_core.h:2150-2172),
// so a co-resident decode request can make it SHRINK mid-stream. `chunks_for` is
// non-increasing in the unit, so recomputing the OLD tail's contribution with a
// SMALLER unit than the one that charged it can exceed what was recorded:
//     work.chunks (= 1 for a 3072-token tail at unit 3072) < chunks_for(3072, 128) (= 24)
// That used to throw std::logic_error("sequence rebuild chunk accounting is invalid")
// and kill the request. A mid-stream unit shrink is a legal, expected event, so it
// must not be an error.
//
// What is recoverable from the arguments: the tail's charge at the CURRENT unit.
// What is not: the prefix term, which is history charged at the units that were in
// force then. Therefore, when the recorded accounting is representable
//     (work.chunks >= chunks_for(old_tail, unit))
// the split is preserved exactly as before, and when it is not, the whole charged
// span [0, frontier) is re-charged at the unit that is actually in force:
//     chunks_for(tail_begin, unit) + chunks_for(frontier - tail_begin, unit)
// This is an upper bound on the true count (a coarser unit never needs MORE units
// for the same span), it is EXACT whenever tail_begin == 0 -- the common case,
// where the prefix is empty and the recomputation is the identity -- and it makes
// the accounting consistent at the current unit from then on, so later advances are
// exact again. Over-charging a prefill cost estimate is the conservative direction:
// under-charging would make a checkpoint look cheaper to rebuild than it is.
inline void advance_segmented_rebuild_work(runtime::PrefillWork& work, std::uint32_t tail_begin,
                                           std::uint32_t previous_frontier, std::uint32_t frontier,
                                           std::uint32_t prefill_chunk) {
    if (frontier < previous_frontier || work.tokens != previous_frontier ||
        tail_begin > previous_frontier || prefill_chunk == 0) {
        throw std::logic_error("sequence rebuild work is not aligned with its frontier");
    }
    if (frontier == previous_frontier) { return; }

    const std::uint64_t old_tail = previous_frontier - tail_begin;
    const std::uint64_t new_tail = frontier - tail_begin;
    const auto chunks_for        = [prefill_chunk](std::uint64_t tokens) {
        return tokens == 0 ? 0ULL : 1ULL + (tokens - 1ULL) / prefill_chunk;
    };
    const std::uint64_t old_tail_chunks = chunks_for(old_tail);
    const std::uint64_t new_tail_chunks = chunks_for(new_tail);

    const bool representable     = work.chunks >= old_tail_chunks;
    const std::uint64_t prefix   = representable ? work.chunks - old_tail_chunks
                                                 : chunks_for(tail_begin);

    const runtime::PrefillWork total = runtime::make_prefill_work(
        0, frontier, work.vision_items, work.vision_patches, prefill_chunk);
    work.chunks          = prefix + new_tail_chunks;
    work.tokens          = total.tokens;
    work.attention_pairs = total.attention_pairs;
}

} // namespace ninfer::targets::qwen3_6::runtime_support
