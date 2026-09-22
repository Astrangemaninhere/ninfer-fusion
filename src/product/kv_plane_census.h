#pragma once

// The PLANE CENSUS of a RESOLVED KV store: how many layers own planes, how many own
// NONE, and how many distinct codecs those planes are cut with.
//
// WHY THIS EXISTS (the two questions item 10 leaves unanswered)
//
// (a) MIXED COMPRESSION. A store can span more than one codec across its layers, and
//     the tree names that state -- recall_codec_admitted(RecallCodec::Mixed)
//     (spec/turn_recall_journal.h:209, admitted 2026-09-18 by measurement) and its own
//     gloss "several codes across the layers of one page" (:138). The only PRODUCER of
//     that predicate in the tree is the target's classifier,
//     ProgramImplCore::turn_recall_codec_of_layers() (program_impl.h:13503-13518), and
//     it answers a NARROWER question: it walks the layers that carry a COLD SLOT
//     (`view.cold_slots.data == nullptr` -> skip) and returns Mixed iff it saw both
//     NVFP4 and I8 among THEM. Its own comment still says "Two codecs among the
//     slot-bearing layers is `Mixed`, which admits nothing" -- which the journal
//     admission has since contradicted -- so that comment cannot be used as the
//     definition either. Neither question is answerable at the product layer today:
//     "which codecs does this store span" has no function, and "is this store mixed"
//     has no function.
//
// (b) INDEPENDENT PLANE DISCARD. A layer can own NO PLANES AT ALL, and the tree names
//     that state too (KvCacheStorage::Dropped = 7, include/ninfer/types.h:47) and
//     publishes it: the memory summary reports `Dropped` for a discarded layer rather
//     than the empty view's default member (program_impl.h:17276-17284, the D5 rule;
//     the view itself is at state/decoder_state.cpp:203). `Dropped` is NOT a codec --
//     it is the absence of both planes -- so a census that counted it as one would
//     report `{dropped, nvfp4}` as a MIXED store, i.e. it would manufacture a
//     two-codec page out of a one-codec store that happens to have a hole in it. That
//     false positive is the specific reading this header refuses, and it is refused by
//     a named branch rather than by the order in which a loop happens to run.
//
// WHAT IS COUNTED, AND WHAT IS DELIBERATELY NOT
//
//   * INPUT is the RESOLVED per-layer table, not an option string. MemorySummary
//     carries exactly (kv_layer_storage, kv_full_attention_layers) as the page pool was
//     actually built from it (include/ninfer/types.h:1166-1173), so a census is a
//     statement about the RUN, not about what was asked for. `--kv-dtype nvfp4` with a
//     registered default table on top of it is a mixed store, and the global dtype
//     alone cannot say so.
//
//   * DISTINCTNESS IS BY DType, not by the KvCacheStorage spelling. Two spellings that
//     name one codec are one codec: Fp8E4M3Row256 and Fp8Group16 both resolve to
//     DType::FP8_E4M3FN (product/kv_storage_dtype.h:54-61), so `{fp8-e4m3-r256,
//     fp8-g16}` is a ONE-codec store. Keying on the raw storage value would report it as
//     mixed -- one of the two loose readings this header exists to refuse. DType is also
//     what the engine's own classifier keys on (program_impl.h:13511 compares
//     `view.dtype`), so this census and that one agree on what a codec IS.
//
//   * EVERY plane-bearing codec must be RESOLVABLE, and the census delegates that to
//     product::kv_dtype_for_storage (the ONE decision point, kv_storage_dtype.h:44),
//     which throws for the fork-survey storages and for a value no enumerator names. A
//     codec the engine cannot build must not be COUNTED as a codec: the count would then
//     describe a store the engine refuses to run, and a report is read precisely when
//     the run is being distrusted. The refusal is inherited rather than re-invented.
//
//   * A layer at or past `layers` is not read at all. The bound is the caller's resolved
//     full-attention count, and a table longer than it is the ordinary case (the table is
//     FAMILY-WIDE, kKvLayerStorageSlots = 64, include/ninfer/types.h:104). A count LARGER
//     than the table is a caller error and is refused instead of read past the end.
//
// THE ONE RELATION TO THE ENGINE'S CLASSIFIER, stated here so it is not discovered as a
// disagreement: this census sees every plane-bearing layer, the engine's sees only the
// cold-slot-bearing ones. Therefore `Mixed` from the engine IMPLIES `mixed` here, and
// the converse does not hold. The registered default table of qwen3.6-27b is the
// witness: E8 on layers {0,1,3,4,6,7} and NVFP4 on the rest
// (include/ninfer/types.h:1166-1171, targets/qwen3_6_27b/impl/variant.cpp:24-25) is a
// MIXED store by this census, while the engine classifies it nvfp4, because the rk4v4
// layers carry no cold slot (program_impl.h:13505-13507). Both statements are true and
// they are about different sets; a report that printed one number for both would be
// wrong about one of them.

#include "ninfer/types.h"
#include "product/kv_storage_dtype.h"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// The bound on the codec set. SEVEN KvCacheStorage values resolve to SIX distinct
// DTypes in the ONE mapping (product/kv_storage_dtype.h:44-...): bf16 -> BF16,
// int8-g64 -> I8, fp8-e4m3-r256 AND fp8-g16 -> FP8_E4M3FN (two spellings, one codec),
// nvfp4-g16 -> NVFP4, iso4e-g16 (KvCacheStorage::Iso3Group16) -> ISO3, rk4v4-g64 ->
// E8Kv. Everything else THROWS there and therefore never reaches the set: Dropped (no
// planes), rk3v4-g64 and rk2v4-g64 (a real tier with no reader -- the writer exists and
// no runtime path calls the decoder), and the five fork-survey storages (no page-pool
// layout in this engine). The bound is set to eight -- six plus headroom for an
// enumerator that lands WITH a codec -- and it is ASSERTED rather than assumed: a ninth
// distinct DType means an enumerator landed without this constant being widened, and
// that is a REFUSAL, not a saturating count. A distinct-codec count that silently stops
// counting is the false negative this whole header is written against, and it would be
// read precisely when a run is being distrusted.
inline constexpr std::size_t kKvPlaneCensusCodecs = 8;

struct KvPlaneCensus {
    std::uint32_t layers          = 0;  // layers examined (the caller's resolved count)
    std::uint32_t plane_bearing   = 0;  // layers that own at least one plane
    std::uint32_t dropped         = 0;  // layers that own NO plane at all
    std::uint32_t distinct_codecs = 0;  // distinct codecs among the PLANE-BEARING layers
    // Meaningful only when distinct_codecs == 1: the single codec of the store. Both
    // halves are kept because they are not interchangeable -- the spelling is what an
    // operator reads, the DType is what the census counted and what the engine's
    // classifier compares.
    KvCacheStorage codec          = KvCacheStorage::Dropped;
    // The product-layer answer to "does this store span more than one codec", i.e. the
    // state the recall journal calls Mixed. It is NOT the engine's RecallCodec: see the
    // relation note above.
    bool mixed                    = false;
};

// `where` is the caller's own provenance for the table, e.g. "kv cache planes". It is
// handed to kv_dtype_for_storage so an unresolvable codec is refused by a message that
// names the caller, not by one that names this header.
[[nodiscard]] inline KvPlaneCensus
kv_plane_census(std::span<const KvCacheStorage> table, std::uint32_t layers,
                std::string_view where) {
    if (layers > table.size()) {
        throw std::invalid_argument(
            std::string(where) + ": census asked for " + std::to_string(layers) +
            " layers but the per-layer KV table holds " + std::to_string(table.size()) +
            "; a count past the end of the table is refused rather than truncated, because "
            "a truncated census reports a smaller store than the run has");
    }
    KvPlaneCensus out;
    out.layers = layers;
    // The DISTINCT set, not a running "last codec seen". Comparing each layer against
    // only the most recent codec counts ALTERNATIONS, not distinct codecs: the
    // registered default table of qwen3.6-27b alternates E8/NVFP4 and would read as
    // ELEVEN codecs under that reading (measured: this is the first shape this function
    // was written in, and the probe's d3_codecs check turned it red). The two readings
    // agree only on stores whose codecs never interleave, which is not a property to
    // rely on.
    std::array<DType, kKvPlaneCensusCodecs> seen{};
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const KvCacheStorage storage = table[layer];
        // (b) A discarded layer owns NO plane of any kind, so it is not a codec. It is
        // counted where it belongs and skipped HERE -- before the codec accounting --
        // so no ordering accident can turn it into a second codec.
        if (storage == KvCacheStorage::Dropped) {
            ++out.dropped;
            continue;
        }
        // Resolvable, or refused by name. A codec the engine has no DType for (the
        // fork-survey storages, or a byte no enumerator names) must not be counted: see
        // the header note. This throws BEFORE the census can return a number.
        const DType dtype = kv_dtype_for_storage(
            storage, std::string(where) + ": layer " + std::to_string(layer));
        bool known = false;
        for (std::uint32_t i = 0; i < out.distinct_codecs; ++i) {
            if (seen[i] == dtype) { known = true; break; }
        }
        if (!known) {
            if (out.distinct_codecs == kKvPlaneCensusCodecs) {
                throw std::invalid_argument(
                    std::string(where) + ": layer " + std::to_string(layer) +
                    " carries a " + std::to_string(out.distinct_codecs + 1) +
                    "th distinct KV codec, past the " +
                    std::to_string(kKvPlaneCensusCodecs) +
                    " this census tracks; the set is full, so a distinct-codec count "
                    "can no longer be exact and is refused instead of reported short");
            }
            if (out.distinct_codecs == 0) { out.codec = storage; }
            seen[out.distinct_codecs] = dtype;
            ++out.distinct_codecs;
        }
        ++out.plane_bearing;
    }
    out.mixed = out.distinct_codecs > 1;
    return out;
}

// The MemorySummary overload: the shape a run actually publishes. There is no second
// definition of the census -- this forwards, so the two cannot drift.
[[nodiscard]] inline KvPlaneCensus kv_plane_census(const MemorySummary& memory,
                                                   std::string_view where) {
    return kv_plane_census(
        std::span<const KvCacheStorage>(memory.kv_layer_storage.data(),
                                        memory.kv_layer_storage.size()),
        memory.kv_full_attention_layers, where);
}

// The operator's line. Named cases, so a reader can key a regex off the FIRST token
// rather than off the absence of a word: "mixed", "single", "no-planes", "empty".
//   empty      the cache has no full-attention layer at all (layers == 0)
//   no-planes  every layer is discarded, so the store has no codec and no planes
//   single     exactly one codec across the plane-bearing layers
//   mixed      two or more codecs across the plane-bearing layers
// The dropped count is always printed, INCLUDING when it is 0: a report that omits the
// zero leaves the reader unable to tell "no layer was discarded" from "this build does
// not report discard".
[[nodiscard]] inline std::string kv_plane_census_line(const KvPlaneCensus& census) {
    const std::string layers  = std::to_string(census.layers);
    const std::string dropped = std::to_string(census.dropped);
    if (census.layers == 0) { return "empty (no full-attention layer)"; }
    if (census.plane_bearing == 0) {
        return "no-planes (" + layers + " full-attention layers, all " + dropped +
               " discarded)";
    }
    const std::string codecs = std::to_string(census.distinct_codecs);
    if (!census.mixed) {
        return "single (" + layers + " full-attention layers, " + codecs + " codec " +
               std::string(kv_operator_token(census.codec)) + ", " + dropped +
               " discarded)";
    }
    return "mixed (" + layers + " full-attention layers, " + codecs + " codecs across " +
           std::to_string(census.plane_bearing) + " plane-bearing layers, " + dropped +
           " discarded)";
}

} // namespace ninfer::product
