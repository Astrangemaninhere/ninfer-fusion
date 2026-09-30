#pragma once

// src/spec/sum_dir_vector.h -- the VECTOR half of the directory over recall blocks.
//
//   Independent feature. EXPERIMENTAL. OFF BY DEFAULT. Host-only, std-only: no CUDA, no engine
//   headers, so it compiles and runs under plain `g++` (tests/test_sum_dir_vector.cpp).
//
// WHAT THIS IS, AND WHERE IT SITS
//   src/spec/sum_dir.h builds a directory whose ROW IS THE RECALL BLOCK and whose only retriever is
//   `SumDir::search_summaries(term)` -- a case-sensitive SUBSTRING test over the catalogue line.
//   That function's own comment names this file's job, verbatim (sum_dir.h:1504-1506):
//
//       "A real retriever (BM25 over the summaries; or an any-position index, which the absent
//        ledger specification argued the medial shape needs) replaces this function's body, not
//        its contract."
//
//   This header IS that replacement body for the vector case: a per-row EMBEDDING column, a
//   cosine/dot ranking over it, and a top-k that returns the SAME type `search_summaries` returns
//   (std::vector<std::size_t>, i.e. row indices) under the SAME offering rule. The two are
//   interchangeable by construction, and that interchangeability is PROVEN in the test rather than
//   asserted here -- section 7 of tests/test_sum_dir_vector.cpp builds a one-hot embedding column
//   whose term axis is the substring the catalogue search looks for, and requires the two
//   retrievers to return the same row set over a whole corpus.
//
// WHAT WAS NOT TOUCHED, AND WHY (the reason this is a new file and not a patch to sum_dir.h)
//   * Not one line of src/spec/sum_dir.h changes. Its row stride (kSumDirRowWireBytes == 80), its
//     48-byte header, its magic `SRD1` and its version byte are all left exactly as they are, so
//     every byte an older build wrote still loads and every byte this build writes still loads in
//     an older one. The vector column has its OWN file and its OWN magic for that reason.
//   * sum_dir.h states the ONE RULE this file must not break, at :89-96: "IDENTITY IS CONTENT, NOT
//     POSITION. A recalled block is rebuilt at a NEW position, so nothing positional may enter
//     `block_identity`". The column therefore stores the row's own `block_identity` next to each
//     vector, and `set()` TAKES that identity from the directory instead of accepting it from the
//     caller -- the same discipline as `SumDir::append()` computing the digest itself
//     (sum_dir.h [text: an API taking a caller-supplied identity; = :1599 on 2026-09-25]: "an API taking a caller-supplied identity would be an API through
//     which position could sneak back in").
//   * `SumDir::sort_rows()` permutes its rows (sum_dir.h:1447-1478) and re-derives every row's
//     `summary_index`. A row-indexed column would silently re-point at that moment. So the column
//     carries the identity per slot and `search()` REFUSES a slot whose stored identity no longer
//     equals its row's -- a stale slot is counted in the report, never used. `rebind()` re-anchors
//     the column by identity after a sort, which is the vector twin of what sort_rows() does to
//     the catalogue column.
//
// WHAT THIS HEADER DOES NOT DO, NAMED SO IT IS NOT MISTAKEN FOR DONE
//   * It does not EMBED anything. There is no tokenizer, no encoder and no model call in this
//     file: a float vector arrives from the caller. Where a vector could come from is a separate
//     decision with a real engine cost (see the report's section 3), and this header is the half
//     that is provably correct once one exists.
//   * It is not wired into the engine. `search_summaries`, `sum_dir_unload_plan` and
//     `sum_dir_idle_eligible` have ZERO callers outside sum_dir.h and tests/test_sum_dir.cpp
//     today; this file does not change that, and it deliberately has no engine-side caller so
//     that landing it CANNOT move the engine binary by one byte.

#include "spec/sum_dir.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::spec::sum_dir::vector_dir {

// ---------------------------------------------------------------------------
// the two strategies, each a COMPLETE answer (the SumDirByteAxisPolicy idiom)
// ---------------------------------------------------------------------------

// Which similarity. `Cosine` stores the row's vector UNIT-NORMALISED, so a score is a cosine in
// [-1, 1] and is invariant to the embedding's scale; `Dot` stores it raw, which is the right choice
// only when the embedding's magnitude is itself meaningful.
enum class SumDirVectorMetric : std::uint8_t {
    Cosine = 0,
    Dot    = 1,
};

// What a row with no stored embedding MEANS.
enum class SumDirVectorMissing : std::uint8_t {
    // DEFAULT. A row with no embedding is NOT OFFERED. Mirrors
    // `kSumDirByteAxisPolicyDefault == RefuseWhenAbsent` (sum_dir.h:578-579): the compatibility
    // setting, so landing this changes no observable behaviour until a caller asks otherwise.
    RefuseWhenMissing = 0,
    // OPT-IN. The row is simply absent from the ranking. The difference between the two is what
    // the report counts in `refused_missing`, so the choice is MEASURED instead of argued about.
    SkipMissing = 1,
};

// The OFFERING rule -- which rows are retrieval targets at all. This is the one knob that decides
// whether this retriever is a drop-in body for `search_summaries` or an extension of it.
//
// WHY `RequireEmbedding` IS THE DEFAULT, and it is a deliberate departure from the
// `SumDirByteAxisPolicy` precedent: there, the default was the setting that "changes no observable
// behaviour until a caller asks for something else" (sum_dir.h [text: PriceWhenAbsent = 1; = :682 on 2026-09-25]), because a caller ALREADY
// existed. Here none does -- the summarizer that writes catalogue lines is the very pass the engine
// does not have (see the report) -- so the compatibility argument does not apply, and defaulting to
// `RequireCatalogue` would re-block vector recall on the model call it is supposed to be free of.
// The default is therefore the setting the feature is FOR: a Live row with an embedding is a
// target. `RequireCatalogue` is kept, and is what section 7 uses to prove the drop-in equality.
enum class SumDirVectorOffer : std::uint8_t {
    // DEFAULT. A `Live` row WITH AN EMBEDDING is a target, catalogue line or not.
    RequireEmbedding = 0,
    // OPT-IN. Exactly `search_summaries`'s rule: the row must be `Live` (sum_dir.h:1511) and must
    // carry a non-empty catalogue line (:1513). This is the setting under which the two retrievers
    // are provably the same function on a one-hot embedding.
    RequireCatalogue = 1,
};

inline constexpr SumDirVectorOffer kSumDirVectorOfferDefault = SumDirVectorOffer::RequireEmbedding;
inline constexpr SumDirVectorMetric kSumDirVectorMetricDefault = SumDirVectorMetric::Cosine;
inline constexpr SumDirVectorMissing kSumDirVectorMissingDefault =
    SumDirVectorMissing::RefuseWhenMissing;
inline constexpr std::uint32_t kSumDirVectorDefaultTopK = 8U;

[[nodiscard]] inline const char* sum_dir_vector_metric_name(SumDirVectorMetric metric) noexcept {
    return metric == SumDirVectorMetric::Cosine ? "cosine" : "dot";
}
[[nodiscard]] inline const char* sum_dir_vector_missing_name(SumDirVectorMissing missing) noexcept {
    return missing == SumDirVectorMissing::RefuseWhenMissing ? "refuse" : "skip";
}
[[nodiscard]] inline const char* sum_dir_vector_offer_name(SumDirVectorOffer offer) noexcept {
    return offer == SumDirVectorOffer::RequireCatalogue ? "require-catalogue" : "require-embedding";
}
// Unknown values are NEVER silently accepted, in either parser: a typo keeps the default and
// reports `recognised = false`, so a misspelling is a visible message and not an invisibly
// different strategy (the same contract as `sum_dir_byte_axis_policy_parse`, sum_dir.h:592-602).
[[nodiscard]] inline SumDirVectorMetric
sum_dir_vector_metric_parse(const char* text, bool& recognised) noexcept {
    recognised = true;
    if (text == nullptr) { recognised = false; return kSumDirVectorMetricDefault; }
    const std::string_view value(text);
    if (value == "cosine") { return SumDirVectorMetric::Cosine; }
    if (value == "dot") { return SumDirVectorMetric::Dot; }
    recognised = false;
    return kSumDirVectorMetricDefault;
}
[[nodiscard]] inline SumDirVectorMissing
sum_dir_vector_missing_parse(const char* text, bool& recognised) noexcept {
    recognised = true;
    if (text == nullptr) { recognised = false; return kSumDirVectorMissingDefault; }
    const std::string_view value(text);
    if (value == "refuse") { return SumDirVectorMissing::RefuseWhenMissing; }
    if (value == "skip") { return SumDirVectorMissing::SkipMissing; }
    recognised = false;
    return kSumDirVectorMissingDefault;
}
[[nodiscard]] inline SumDirVectorOffer
sum_dir_vector_offer_parse(const char* text, bool& recognised) noexcept {
    recognised = true;
    if (text == nullptr) { recognised = false; return kSumDirVectorOfferDefault; }
    const std::string_view value(text);
    if (value == "require-catalogue") { return SumDirVectorOffer::RequireCatalogue; }
    if (value == "require-embedding") { return SumDirVectorOffer::RequireEmbedding; }
    recognised = false;
    return kSumDirVectorOfferDefault;
}

// ---------------------------------------------------------------------------
// knobs -- default OFF, on this feature family's own env prefix
// ---------------------------------------------------------------------------
//
// `dim` has NO default on purpose. An embedding width silently defaulted to some model's hidden
// size is exactly the kind of "state with no check behind it" this project has already paid for, so
// an unset `NINFER_SUM_DIR_VEC_DIM` leaves `enabled == false` and sets `dim_missing`.
struct SumDirVectorKnobs {
    bool                 enabled  = false; // NINFER_SUM_DIR_VEC=1 (or any non-"0" value)
    bool                 dim_missing = false;
    std::uint32_t        dim      = 0;     // NINFER_SUM_DIR_VEC_DIM, required
    std::uint32_t        top_k    = kSumDirVectorDefaultTopK; // NINFER_SUM_DIR_VEC_TOPK
    SumDirVectorMetric   metric   = kSumDirVectorMetricDefault;
    SumDirVectorMissing  missing  = kSumDirVectorMissingDefault;
    SumDirVectorOffer    offer    = kSumDirVectorOfferDefault;
    bool                 metric_error  = false;
    bool                 missing_error = false;
    bool                 offer_error   = false;
    bool                 top_k_error   = false;

    [[nodiscard]] static SumDirVectorKnobs from_env() {
        SumDirVectorKnobs knobs;
        if (const char* enable = std::getenv("NINFER_SUM_DIR_VEC")) {
            if (enable[0] != '\0' && enable[0] != '0') { knobs.enabled = true; }
        }
        if (const char* dim = std::getenv("NINFER_SUM_DIR_VEC_DIM")) {
            const long value = std::atol(dim);
            if (value > 0) { knobs.dim = static_cast<std::uint32_t>(value); } else { knobs.dim_missing = true; }
        } else {
            knobs.dim_missing = true;
        }
        if (const char* topk = std::getenv("NINFER_SUM_DIR_VEC_TOPK")) {
            const long value = std::atol(topk);
            if (value > 0) { knobs.top_k = static_cast<std::uint32_t>(value); } else { knobs.top_k_error = true; }
        }
        if (const char* metric = std::getenv("NINFER_SUM_DIR_VEC_METRIC")) {
            knobs.metric = sum_dir_vector_metric_parse(metric, knobs.metric_error);
        }
        if (const char* missing = std::getenv("NINFER_SUM_DIR_VEC_MISSING")) {
            knobs.missing = sum_dir_vector_missing_parse(missing, knobs.missing_error);
        }
        if (const char* offer = std::getenv("NINFER_SUM_DIR_VEC_OFFER")) {
            knobs.offer = sum_dir_vector_offer_parse(offer, knobs.offer_error);
        }
        // An armed vector retriever with no width is not armed: refuse it rather than guess.
        if (knobs.enabled && knobs.dim == 0) { knobs.enabled = false; }
        return knobs;
    }
};

// ---------------------------------------------------------------------------
// the setter's verdict, and the reader's accounting
// ---------------------------------------------------------------------------

// `set()` reports a NAMED refusal instead of storing something a reader would have to re-diagnose.
enum class SumDirVectorSetResult : std::uint8_t {
    Stored              = 0,
    RefusedDimMismatch  = 1, // the vector's width is not the column's
    RefusedNotFinite    = 2, // a NaN or an infinity is present: it would poison every dot product
    RefusedZeroNorm     = 3, // a cosine against a zero vector is undefined, not 0
    RefusedRowOutOfRange = 4,
};

[[nodiscard]] inline const char* sum_dir_vector_set_result_name(SumDirVectorSetResult r) noexcept {
    switch (r) {
    case SumDirVectorSetResult::Stored: return "stored";
    case SumDirVectorSetResult::RefusedDimMismatch: return "refused-dim-mismatch";
    case SumDirVectorSetResult::RefusedNotFinite: return "refused-not-finite";
    case SumDirVectorSetResult::RefusedZeroNorm: return "refused-zero-norm";
    case SumDirVectorSetResult::RefusedRowOutOfRange: return "refused-row-out-of-range";
    }
    return "unknown";
}

// One ranked hit. `identity` is the row's CONTENT identity, carried alongside the score so a caller
// can bind the answer to the block rather than to the row it happened to sit at -- which is what
// makes an answer survive a `sort_rows()`.
struct SumDirVectorHit {
    std::size_t  row      = 0;
    float        score    = 0.0f; // cosine in [-1, 1] under Cosine; the raw dot under Dot
    SumDirDigest identity{};
};

// Every number a caller needs to tell "no hit" from "not searched". `refused_missing` is what the
// `SumDirVectorMissing` strategy decides; `refused_stale` is the way a slot can be present but not
// trustworthy (a `sort_rows()` moved its row and nobody called `rebind()`); `refused_not_live` and
// `refused_no_catalogue` are `search_summaries`'s own two gates, counted so the drop-in body cannot
// silently lose a row the substring retriever would have found.
struct SumDirVectorSearchReport {
    std::uint32_t rows_total        = 0;
    std::uint32_t considered        = 0; // rows that were actually scored
    std::uint32_t hits              = 0; // <= top_k
    std::uint32_t refused_missing   = 0; // no embedding stored for this row
    std::uint32_t refused_stale     = 0; // stored identity != the row's identity (a sort happened)
    std::uint32_t refused_not_live  = 0; // state != Live -- search_summaries refuses these too
    std::uint32_t refused_no_catalogue = 0; // no non-empty line, under RequireCatalogue
    std::uint32_t refused_query     = 0; // 1 iff the QUERY itself was unusable
    std::uint32_t top_k             = 0;
    bool          truncated         = false; // more than top_k rows were scored
};

// ---------------------------------------------------------------------------
// wire format -- ITS OWN FILE, ITS OWN MAGIC, sum_dir.h's v1 format untouched
// ---------------------------------------------------------------------------
//
// Little-endian, host-independent: a fixed 48-byte header, then the slots at one fixed 40-byte
// stride, then the flat float payload. The seal is sum_dir.h's own reflected CRC-32, reached
// through `sum_dir_crc32` -- so the two artefacts are checkable by one tool and there is no second
// polynomial in the tree.
//
// `dim` and the three strategy values are IN the header, so a reader that disagrees with the writer
// refuses instead of reinterpreting: the same discipline as sum_dir.h's version and granularity
// checks (:1622-1629). And the header carries a `directory_payload_digest`, the digest of the
// SumDir payload these vectors were taken against: a column attached to a DIFFERENT directory is a
// refusal, which is what makes `refused_stale` in the report rare rather than the only defence.
inline constexpr std::uint32_t kSumDirVectorMagic        = 0x31445653U; // 'S','V','D','1'
inline constexpr std::uint8_t  kSumDirVectorVersion      = 1U;
inline constexpr std::uint32_t kSumDirVectorHeaderBytes  = 48U;
// The seal covers [0, 36) -- NOT [0, 44). It must not cover its own field at [36, 40), or
// serialize (which hashes while that field is still zero) and deserialize (which hashes with
// the field already written) can never agree and every load refuses. The struct comment at
// SumDirVectorHeader::header_digest already documents [0, 36).
inline constexpr std::uint32_t kSumDirVectorHeaderCovered = 36U;
inline constexpr std::uint32_t kSumDirVectorSlotBytes    = 40U;

struct SumDirVectorHeader {
    std::uint32_t magic        = kSumDirVectorMagic; //  0
    std::uint8_t  version      = kSumDirVectorVersion; // 4
    std::uint8_t  metric       = 0;                  //  5
    std::uint8_t  missing      = 0;                  //  6
    std::uint8_t  offer        = 0;                  //  7
    std::uint32_t dim          = 0;                  //  8
    std::uint32_t top_k        = 0;                  // 12
    std::uint64_t slot_count   = 0;                  // 16
    std::uint32_t directory_payload_digest = 0;      // 24
    std::uint32_t slot_bytes   = 0;                  // 28  slot_count * dim * 4, explicit
    std::uint32_t payload_digest = 0;                // 32  crc over slots ++ floats
    std::uint32_t header_digest  = 0;                // 36  crc over [0, 36)
    std::uint32_t reserved0      = 0;                // 40  must be 0
    std::uint32_t reserved1      = 0;                // 44  must be 0
};
static_assert(sizeof(SumDirVectorHeader) == kSumDirVectorHeaderBytes,
              "the vector header is one fixed unit; a packed reader must see the same 48");

struct SumDirVectorLoadReport {
    std::uint64_t slots_read = 0;
    std::uint32_t dim        = 0;
    bool header_ok  = false;
    bool payload_ok = false;
    bool truncated  = false;
    std::string error;
};

// ---------------------------------------------------------------------------
// the column
// ---------------------------------------------------------------------------
// One slot per DIRECTORY ROW, in row order -- the same lockstep invariant the catalogue column
// keeps ("summaries_.size() == rows_.size() always", sum_dir.h:425-433), for the same reason: a slot
// that a loader cannot accept must be droppable without disturbing any other row's vector. A slot
// with `dim == 0` is the empty slot; `has()` is what asks.
struct SumDirVectorSlot {
    SumDirDigest  identity{};          // the row's CONTENT identity at set() time -- not a position
    std::uint32_t dim    = 0;          // 0 == empty
    std::uint32_t offset = 0;          // into the flat store
};

class SumDirVectorColumn {
public:
    SumDirVectorColumn() = default;

    explicit SumDirVectorColumn(std::uint32_t dim,
                               SumDirVectorMetric metric = kSumDirVectorMetricDefault,
                               SumDirVectorMissing missing = kSumDirVectorMissingDefault,
                               SumDirVectorOffer offer = kSumDirVectorOfferDefault)
        : dim_(dim), metric_(metric), missing_(missing), offer_(offer) {
        if (dim_ == 0) { throw std::invalid_argument("sum_dir vector: dim 0 is not a column"); }
    }

    [[nodiscard]] std::uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] SumDirVectorMetric metric() const noexcept { return metric_; }
    [[nodiscard]] SumDirVectorMissing missing() const noexcept { return missing_; }
    [[nodiscard]] SumDirVectorOffer offer() const noexcept { return offer_; }
    [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }
    [[nodiscard]] bool has(std::size_t row) const noexcept {
        return row < slots_.size() && slots_[row].dim != 0;
    }
    [[nodiscard]] const std::vector<SumDirVectorSlot>& slots() const noexcept { return slots_; }
    [[nodiscard]] const std::vector<float>& store() const noexcept { return store_; }
    // The whole reason slots carry an identity: a slot whose identity has moved is a slot this
    // column must not answer with.
    [[nodiscard]] std::uint32_t stale_slots(const SumDir& directory) const noexcept {
        std::uint32_t stale = 0;
        for (std::size_t row = 0; row < slots_.size() && row < directory.rows().size(); ++row) {
            if (slots_[row].dim == 0) { continue; }
            if (!(slots_[row].identity == directory.rows()[row].block_identity)) { ++stale; }
        }
        return stale;
    }

    // THE ONE SETTER. Three things are deliberately not the caller's to choose:
    //   * the IDENTITY. It is read from `directory.rows()[row].block_identity`, so two copies of one
    //     block at different offsets receive the SAME key and are one target, which is the whole
    //     content-not-position rule (sum_dir.h:89-96) applied to the vector case.
    //   * the NORMALISATION. Under `Cosine` the vector is stored unit-length, so a caller cannot
    //     change the ranking by scaling its encoder.
    //   * whether an unusable vector is stored at all. It is not, and the reason is returned.
    [[nodiscard]] SumDirVectorSetResult set(const SumDir& directory, std::size_t row,
                                            const float* embedding, std::uint32_t dim) {
        if (row >= directory.rows().size()) { return SumDirVectorSetResult::RefusedRowOutOfRange; }
        if (embedding == nullptr || dim != dim_) { return SumDirVectorSetResult::RefusedDimMismatch; }
        float norm2 = 0.0f;
        for (std::uint32_t k = 0; k < dim_; ++k) {
            const float value = embedding[k];
            if (!std::isfinite(value)) { return SumDirVectorSetResult::RefusedNotFinite; }
            norm2 = std::fma(value, value, norm2);
        }
        if (!(norm2 > 0.0f)) { return SumDirVectorSetResult::RefusedZeroNorm; }
        // The normalisation buffer is sized ONCE and never shrunk: `set()` is called per row on the
        // idle path, and a per-call vector here would be an allocation per row for no reason.
        if (scratch_.size() != dim_) { scratch_.assign(dim_, 0.0f); }
        if (metric_ == SumDirVectorMetric::Cosine && !(std::fabs(norm2 - 1.0f) <= 1e-6f)) {
            const float inverse = 1.0f / std::sqrt(norm2);
            for (std::uint32_t k = 0; k < dim_; ++k) {
                scratch_[k] = embedding[k] * inverse;
            }
        } else {
            for (std::uint32_t k = 0; k < dim_; ++k) { scratch_[k] = embedding[k]; }
        }
        if (slots_.size() <= row) { slots_.resize(row + 1U); }
        SumDirVectorSlot& slot = slots_[row];
        slot.identity = directory.rows()[row].block_identity; // from the DIRECTORY, never the caller
        slot.dim      = dim_;
        slot.offset   = static_cast<std::uint32_t>(store_.size());
        store_.insert(store_.end(), scratch_.begin(), scratch_.begin() + dim_);
        return SumDirVectorSetResult::Stored;
    }

    [[nodiscard]] SumDirVectorSetResult set(const SumDir& directory, std::size_t row,
                                            const std::vector<float>& embedding) {
        return set(directory, row, embedding.data(), static_cast<std::uint32_t>(embedding.size()));
    }

    // The vector twin of what `sort_rows()` does to the catalogue column: after the directory is
    // re-sorted, a slot that still matches a row by IDENTITY is re-anchored to that row's new index.
    // A slot whose identity is gone (the block was dropped by a loader) is CLEARED, not left
    // pointing at whoever now sits at the old index.
    void rebind(const SumDir& directory) {
        std::vector<SumDirVectorSlot> rebound(directory.rows().size());
        for (SumDirVectorSlot& slot : slots_) {
            if (slot.dim == 0) { continue; }
            for (std::size_t row = 0; row < directory.rows().size(); ++row) {
                if (!(directory.rows()[row].block_identity == slot.identity)) { continue; }
                if (rebound[row].dim != 0) { continue; } // first match wins, deterministically
                rebound[row] = slot;
                break;
            }
        }
        slots_.swap(rebound);
        rebind_count_ = rebind_count_ + 1U;
    }
    [[nodiscard]] std::uint32_t rebind_count() const noexcept { return rebind_count_; }

    // THE RETRIEVAL. `search_summaries(term)`'s body, replaced -- same return type, same offering
    // rule under `RequireCatalogue`, and the ranking is (score DESC, row index ASC), so the order is
    // total and byte-for-byte reproducible: two runs on the same inputs cannot disagree.
    [[nodiscard]] std::vector<SumDirVectorHit>
    search(const SumDir& directory, const float* query,
           SumDirVectorSearchReport* report = nullptr) const {
        return search_top_k(directory, query, top_k_default_, report);
    }

    [[nodiscard]] std::vector<SumDirVectorHit>
    search_top_k(const SumDir& directory, const float* query, std::uint32_t top_k,
                 SumDirVectorSearchReport* report = nullptr) const {
        SumDirVectorSearchReport local;
        local.top_k      = top_k;
        local.rows_total = static_cast<std::uint32_t>(directory.rows().size());

        // The QUERY is judged once, before any row: a query that cannot be scored is one refusal,
        // not `rows_total` of them.
        float query_norm2 = 0.0f;
        bool query_ok = query != nullptr;
        if (query_ok) {
            for (std::uint32_t k = 0; k < dim_; ++k) {
                const float value = query[k];
                if (!std::isfinite(value)) { query_ok = false; break; }
                query_norm2 = std::fma(value, value, query_norm2);
            }
            if (metric_ == SumDirVectorMetric::Cosine && !(query_norm2 > 0.0f)) { query_ok = false; }
        }
        if (!query_ok) {
            local.refused_query = 1U;
            if (report != nullptr) { *report = local; }
            return {};
        }
        std::vector<float> query_unit;
        const float* query_values = query;
        if (metric_ == SumDirVectorMetric::Cosine) {
            const float inverse = 1.0f / std::sqrt(query_norm2);
            query_unit.resize(dim_);
            for (std::uint32_t k = 0; k < dim_; ++k) { query_unit[k] = query[k] * inverse; }
            query_values = query_unit.data();
        }

        std::vector<SumDirVectorHit> hits;
        for (std::size_t row = 0; row < directory.rows().size(); ++row) {
            const SumDirRow& directory_row = directory.rows()[row];
            // GATE 1, `search_summaries`'s own (`:1511`): a row that is not Live is not a target.
            if (directory_row.state != SumDirState::Live) { ++local.refused_not_live; continue; }
            // GATE 2, if the caller asked for the drop-in body (`:1513`).
            if (offer_ == SumDirVectorOffer::RequireCatalogue &&
                !sum_dir_row_has_summary(directory_row)) {
                ++local.refused_no_catalogue;
                continue;
            }
            // GATE 3: is there a vector for this row, and is it still this row's?
            if (row >= slots_.size() || slots_[row].dim == 0) {
                if (missing_ == SumDirVectorMissing::RefuseWhenMissing) {
                    ++local.refused_missing;
                    if (report != nullptr) { *report = local; }
                    return {}; // RefuseWhenMissing is a REFUSAL, not a shorter list
                }
                ++local.refused_missing;
                continue;
            }
            const SumDirVectorSlot& slot = slots_[row];
            if (!(slot.identity == directory_row.block_identity)) {
                // A slot that survives a sort without a rebind() would answer for the wrong block.
                ++local.refused_stale;
                if (missing_ == SumDirVectorMissing::RefuseWhenMissing) {
                    if (report != nullptr) { *report = local; }
                    return {};
                }
                continue;
            }
            const float* values = store_.data() + slot.offset;
            float score = 0.0f;
            for (std::uint32_t k = 0; k < dim_; ++k) {
                score = std::fma(values[k], query_values[k], score);
            }
            // Unreachable by construction -- `set()` refuses a non-finite vector and the query was
            // judged above -- but a defensive branch must obey the SAME strategy as the others, or
            // the two settings would differ on the one path nobody tests.
            if (!std::isfinite(score)) {
                ++local.refused_missing;
                if (missing_ == SumDirVectorMissing::RefuseWhenMissing) {
                    if (report != nullptr) { *report = local; }
                    return {};
                }
                continue;
            }
            if (metric_ == SumDirVectorMetric::Cosine) {
                // The two unit vectors make |cos| <= 1; clamp so fp noise cannot produce 1.0000001
                // and make two equal vectors compare unequal.
                score = std::fmin(1.0f, std::fmax(-1.0f, score));
            }
            ++local.considered;
            SumDirVectorHit hit;
            hit.row      = row;
            hit.score    = score;
            hit.identity = slot.identity;
            hits.push_back(hit);
        }

        // A TOTAL order, so the ranking is reproducible: score descending, then row ascending. No
        // comparator here consults anything positional about the BLOCK -- `row` is how the caller
        // finds the row again, which is what sum_dir.h:92-94 calls bookkeeping.
        const auto better = [](const SumDirVectorHit& a, const SumDirVectorHit& b) noexcept {
            if (a.score != b.score) { return a.score > b.score; }
            return a.row < b.row;
        };
        if (hits.size() > top_k) {
            std::partial_sort(hits.begin(), hits.begin() + top_k, hits.end(), better);
            local.truncated = true;
            hits.resize(top_k);
        } else {
            std::sort(hits.begin(), hits.end(), better);
        }
        local.hits = static_cast<std::uint32_t>(hits.size());
        if (report != nullptr) { *report = local; }
        return hits;
    }

    [[nodiscard]] std::vector<SumDirVectorHit>
    search_top_k(const SumDir& directory, const std::vector<float>& query, std::uint32_t top_k,
                 SumDirVectorSearchReport* report = nullptr) const {
        if (query.size() != dim_) {
            SumDirVectorSearchReport local;
            local.refused_query = 1U;
            local.top_k         = top_k;
            local.rows_total    = static_cast<std::uint32_t>(directory.rows().size());
            if (report != nullptr) { *report = local; }
            return {};
        }
        return search_top_k(directory, query.data(), top_k, report);
    }

    [[nodiscard]] std::uint32_t top_k_default() const noexcept { return top_k_default_; }
    void set_top_k_default(std::uint32_t top_k) noexcept {
        top_k_default_ = top_k == 0 ? kSumDirVectorDefaultTopK : top_k;
    }

    // -----------------------------------------------------------------------
    // serialization -- a SEPARATE artefact, so sum_dir.h's v1 bytes never move
    // -----------------------------------------------------------------------

    // Slots ++ floats, one digest over the payload the header describes, so a loader can separate
    // "the slots and the vectors agree" from "they do not" -- sum_dir.h's own split (:1540-1541).
    [[nodiscard]] std::uint32_t payload_digest() const noexcept {
        std::uint32_t crc = 0xFFFFFFFFU;
        for (const SumDirVectorSlot& slot : slots_) {
            std::uint8_t wire[kSumDirVectorSlotBytes] = {};
            detail::sum_dir_pack_u64_at(wire, 0, slot.identity.lo);
            detail::sum_dir_pack_u64_at(wire, 8, slot.identity.hi);
            detail::sum_dir_pack_u32_at(wire, 16, slot.dim);
            detail::sum_dir_pack_u32_at(wire, 20, slot.offset);
            detail::sum_dir_pack_u32_at(wire, 24, 0);
            detail::sum_dir_pack_u32_at(wire, 28, 0);
            detail::sum_dir_pack_u32_at(wire, 32, 0);
            detail::sum_dir_pack_u32_at(wire, 36, 0);
            crc = sum_dir_crc32(crc, wire, kSumDirVectorSlotBytes);
        }
        for (const float value : store_) {
            std::uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value), "f32 is 4 bytes");
            std::memcpy(&bits, &value, sizeof(bits));
            std::uint8_t word[4];
            detail::sum_dir_pack_u32_at(word, 0, bits);
            crc = sum_dir_crc32(crc, word, 4);
        }
        return ~crc;
    }

    // `directory` is passed IN and only for its payload digest: it is recorded so a column that is
    // loaded against a different directory is a REFUSAL at load time instead of a wrong answer at
    // search time.
    [[nodiscard]] std::vector<std::uint8_t> serialize(const SumDir& directory) const {
        std::vector<std::uint8_t> out;
        out.reserve(kSumDirVectorHeaderBytes + slots_.size() * kSumDirVectorSlotBytes +
                    store_.size() * 4U);

        std::uint8_t header[kSumDirVectorHeaderBytes] = {};
        detail::sum_dir_pack_u32_at(header, 0, kSumDirVectorMagic);
        header[4] = kSumDirVectorVersion;
        header[5] = static_cast<std::uint8_t>(metric_);
        header[6] = static_cast<std::uint8_t>(missing_);
        header[7] = static_cast<std::uint8_t>(offer_);
        detail::sum_dir_pack_u32_at(header, 8, dim_);
        detail::sum_dir_pack_u32_at(header, 12, top_k_default_);
        detail::sum_dir_pack_u64_at(header, 16, slots_.size());
        detail::sum_dir_pack_u32_at(header, 24, directory.payload_digest());
        detail::sum_dir_pack_u32_at(header, 28, static_cast<std::uint32_t>(store_.size() * 4U));
        detail::sum_dir_pack_u32_at(header, 32, payload_digest());
        std::uint32_t seal = sum_dir_crc32(0xFFFFFFFFU, header, kSumDirVectorHeaderCovered);
        seal = ~seal;
        detail::sum_dir_pack_u32_at(header, 36, seal);

        detail::sum_dir_pack_bytes(out, header, kSumDirVectorHeaderBytes);
        for (const SumDirVectorSlot& slot : slots_) {
            std::uint8_t wire[kSumDirVectorSlotBytes] = {};
            detail::sum_dir_pack_u64_at(wire, 0, slot.identity.lo);
            detail::sum_dir_pack_u64_at(wire, 8, slot.identity.hi);
            detail::sum_dir_pack_u32_at(wire, 16, slot.dim);
            detail::sum_dir_pack_u32_at(wire, 20, slot.offset);
            detail::sum_dir_pack_bytes(out, wire, kSumDirVectorSlotBytes);
        }
        for (const float value : store_) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            std::uint8_t word[4];
            detail::sum_dir_pack_u32_at(word, 0, bits);
            detail::sum_dir_pack_bytes(out, word, 4);
        }
        return out;
    }

    // STRICT by design, exactly as `SumDir::deserialize` is (:1606-1609): an unknown version, an
    // unknown strategy value, a reserved bit that is set, a digest that does not match, a slot count
    // that disagrees with the header's own byte count, or a directory that is not the one these
    // vectors were taken against is a REFUSAL -- an empty column plus a report, never a partial
    // load. A half-loaded column would be worse than none, because the absent slots would read as
    // "this block has no description" rather than "this block was not read".
    [[nodiscard]] static SumDirVectorColumn
    deserialize(const SumDir& directory, const std::uint8_t* data, std::size_t bytes,
                SumDirVectorLoadReport& report) {
        SumDirVectorColumn column;
        if (data == nullptr || bytes < kSumDirVectorHeaderBytes) {
            report.error     = "shorter than one header";
            report.truncated = true;
            return column;
        }
        if (detail::sum_dir_unpack_u32(data, 0) != kSumDirVectorMagic) {
            report.error = "bad magic";
            return column;
        }
        if (data[4] != kSumDirVectorVersion) {
            report.error = "unsupported version";
            return column;
        }
        if (data[5] > static_cast<std::uint8_t>(SumDirVectorMetric::Dot) ||
            data[6] > static_cast<std::uint8_t>(SumDirVectorMissing::SkipMissing) ||
            data[7] > static_cast<std::uint8_t>(SumDirVectorOffer::RequireCatalogue)) {
            report.error = "strategy byte out of range";
            return column;
        }
        if (detail::sum_dir_unpack_u32(data, 40) != 0 || detail::sum_dir_unpack_u32(data, 44) != 0) {
            report.error = "reserved header bits are set";
            return column;
        }
        std::uint32_t header_seal = sum_dir_crc32(0xFFFFFFFFU, data, kSumDirVectorHeaderCovered);
        header_seal = ~header_seal;
        if (header_seal != detail::sum_dir_unpack_u32(data, 36)) {
            report.error = "header digest mismatch";
            return column;
        }
        report.header_ok = true;

        const std::uint32_t dim        = detail::sum_dir_unpack_u32(data, 8);
        const std::uint64_t slot_count = detail::sum_dir_unpack_u64(data, 16);
        const std::uint32_t want_bytes = detail::sum_dir_unpack_u32(data, 28);
        const std::uint32_t directory_digest = detail::sum_dir_unpack_u32(data, 24);
        if (dim == 0) {
            report.error = "dim 0 is not a column";
            return column;
        }
        // THE BINDING TO THE DIRECTORY. A column and a directory that do not describe the same
        // blocks must never be combined; the row's identity check at search time is the second line
        // of defence, not the first.
        if (directory_digest != directory.payload_digest()) {
            report.error = "the vectors were taken against a different directory";
            return column;
        }
        const std::uint64_t expected_slot_bytes = slot_count * static_cast<std::uint64_t>(dim) * 4ULL;
        if (static_cast<std::uint64_t>(want_bytes) != expected_slot_bytes) {
            report.error = "slot byte count disagrees with the header's own slot count";
            return column;
        }
        const std::uint64_t payload_bytes =
            slot_count * kSumDirVectorSlotBytes + expected_slot_bytes;
        if (payload_bytes > bytes - kSumDirVectorHeaderBytes) {
            report.error     = "payload shorter than the header describes";
            report.truncated = true;
            return column;
        }
        const std::uint8_t* payload = data + kSumDirVectorHeaderBytes;
        std::uint32_t payload_seal =
            sum_dir_crc32(0xFFFFFFFFU, payload, static_cast<std::size_t>(payload_bytes));
        payload_seal = ~payload_seal;
        if (payload_seal != detail::sum_dir_unpack_u32(data, 32)) {
            report.error = "payload digest mismatch";
            return column;
        }
        report.payload_ok = true;

        column.dim_           = dim;
        column.metric_        = static_cast<SumDirVectorMetric>(data[5]);
        column.missing_       = static_cast<SumDirVectorMissing>(data[6]);
        column.offer_         = static_cast<SumDirVectorOffer>(data[7]);
        column.top_k_default_ = detail::sum_dir_unpack_u32(data, 12) == 0
                                    ? kSumDirVectorDefaultTopK
                                    : detail::sum_dir_unpack_u32(data, 12);

        const std::uint8_t* float_base =
            payload + static_cast<std::size_t>(slot_count) * kSumDirVectorSlotBytes;
        column.slots_.resize(static_cast<std::size_t>(slot_count));
        column.store_.resize(static_cast<std::size_t>(expected_slot_bytes / 4ULL));
        for (std::size_t index = 0; index < column.slots_.size(); ++index) {
            const std::uint8_t* wire = payload + index * kSumDirVectorSlotBytes;
            SumDirVectorSlot& slot   = column.slots_[index];
            slot.identity.lo = detail::sum_dir_unpack_u64(wire, 0);
            slot.identity.hi = detail::sum_dir_unpack_u64(wire, 8);
            slot.dim         = detail::sum_dir_unpack_u32(wire, 16);
            slot.offset      = detail::sum_dir_unpack_u32(wire, 20);
            // The slot's own bounds are checked against the STORE, so a torn or invented offset
            // cannot make `search()` read outside the flat array.
            if (slot.dim != 0) {
                if (slot.dim != dim ||
                    static_cast<std::uint64_t>(slot.offset) + dim > column.store_.size()) {
                    report.error = "slot is out of range of its own store";
                    column.slots_.clear();
                    column.store_.clear();
                    return column;
                }
            }
        }
        for (std::size_t index = 0; index < column.store_.size(); ++index) {
            const std::uint32_t bits =
                detail::sum_dir_unpack_u32(float_base, index * 4U);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            column.store_[index] = value;
        }
        report.slots_read = static_cast<std::uint64_t>(column.slots_.size());
        report.dim        = dim;
        return column;
    }

    [[nodiscard]] static SumDirVectorColumn
    deserialize(const SumDir& directory, const std::vector<std::uint8_t>& bytes,
                SumDirVectorLoadReport& report) {
        return deserialize(directory, bytes.data(), bytes.size(), report);
    }

private:
    std::uint32_t       dim_ = 0;
    SumDirVectorMetric  metric_  = kSumDirVectorMetricDefault;
    SumDirVectorMissing missing_ = kSumDirVectorMissingDefault;
    SumDirVectorOffer   offer_   = kSumDirVectorOfferDefault;
    std::uint32_t       top_k_default_ = kSumDirVectorDefaultTopK;
    std::uint32_t       rebind_count_  = 0;
    std::vector<SumDirVectorSlot> slots_;
    std::vector<float>  store_;   // flat, dim_ floats per non-empty slot
    std::vector<float>  scratch_; // the normalisation buffer, kept so set() can run in one pass
};

// ---------------------------------------------------------------------------
// THE CONTRACT-PRESERVING FREE FUNCTION -- what a caller swaps `search_summaries` for
// ---------------------------------------------------------------------------
//
// The same signature SHAPE and the same return type as `SumDir::search_summaries` (a
// std::vector<std::size_t> of row indices), so the call site does not change shape when the
// retriever does. It returns the rows only, in the same (score DESC, row ASC) order.
[[nodiscard]] inline std::vector<std::size_t>
sum_dir_vector_search_rows(const SumDir& directory, const SumDirVectorColumn& column,
                           const std::vector<float>& query, std::uint32_t top_k,
                           SumDirVectorSearchReport* report = nullptr) {
    const std::vector<SumDirVectorHit> hits =
        column.search_top_k(directory, query, top_k, report);
    std::vector<std::size_t> rows;
    rows.reserve(hits.size());
    for (const SumDirVectorHit& hit : hits) { rows.push_back(hit.row); }
    return rows;
}

} // namespace ninfer::spec::sum_dir::vector_dir
