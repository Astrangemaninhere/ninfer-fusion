#pragma once

// shard_plan.h -- the multi-device shard arithmetic, the per-rank persistence naming, and
// the ONE predicate the KV policy needs before any of it is safe: whether a page's
// residency bit can still be a single bit once the KV cache is split across cards.
//
// Host-only by construction: no CUDA header, no device query, no collective. Every
// arithmetic and every naming rule here is a pure function, so it is exercisable on the
// single-GPU host this was written on. What is NOT here, and is named as missing rather
// than stubbed, is the collective itself: see "WHAT IS MISSING" at the bottom.
//
// WHY THIS FILE EXISTS AT ALL
// ---------------------------
// The engine today has exactly one device:
//   * `cudaGetDeviceCount` / `cudaSetDevice` occur only in src/core/device.cu, and its
//     only job is to pick one card and bind the thread to it;
//   * there is no rank, no world size, no collective, no peer access, no NCCl, and no
//     shard of anything. (Bounded probe, tracked files under src/ and apps/: the tokens
//     `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeer`, `nccl`, `ncclUniqueId`,
//     `DeviceCount`, `num_devices`, `tensor_parallel`, `tp_rank`, `all_reduce` have ZERO
//     hits; `shard` hits only src/ops/split_attention.cu, which is a POSITION split of a
//     single attention's token range inside one process -- unrelated.)
// So the pieces below are the ones that must exist BEFORE a collective is worth writing,
// and every one of them is checkable without a second GPU.
//
// THE GRANULARITY CONTRACT THIS FILE DEFENDS
// ------------------------------------------
// The KV policy's unit is declared in two places, verbatim:
//   * src/targets/qwen3_6/impl/runtime/cold_host_tier.h:
//       "GRANULARITY: one logical page ACROSS ALL TEXT LAYERS, never a slice of one."
//       "... the tier therefore has no per-layer and no per-head eviction notion at all:
//        one page is one residency bit"
//   * src/spec/turn_recall_journal.h:
//       "GRANULARITY: one logical page = kRecallPageTokens (64) tokens ACROSS ALL TEXT
//        LAYERS. Never per layer, never per head, never per token"
//       static_assert(!recall_codec_admitted(RecallCodec::Mixed),
//                     "a page whose layers use different codecs has no single stride")
// This header takes those as the spec and asks the only question multi-device raises:
// which of the two axes (heads, layers) keeps them, and which breaks them.

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 1. Rank / world shape
// ---------------------------------------------------------------------------

// Deliberately ONE axis, not a mesh. A 2-D (tp x pp) world can be added by composing two
// WorldShapes; declaring it now would put an unexercised degree of freedom in the type.
enum class ParallelAxis : std::uint8_t {
    None = 0, // one device; every field below must be the identity
    Tensor,   // KV heads and weight N are split; TEXT LAYERS stay whole on every rank
    Pipeline, // TEXT LAYERS are split; KV heads stay whole on every stage
};

[[nodiscard]] inline std::string_view axis_name(ParallelAxis axis) noexcept {
    switch (axis) {
    case ParallelAxis::None: return "none";
    case ParallelAxis::Tensor: return "tp";
    case ParallelAxis::Pipeline: return "pp";
    }
    return "unknown-axis";
}

struct WorldShape {
    std::uint32_t world_size = 1;
    std::uint32_t rank       = 0;
    ParallelAxis axis        = ParallelAxis::None;

    friend bool operator==(const WorldShape&, const WorldShape&) = default;
};

// The identity check, and the reason it is a free function rather than an assert: the
// engine must be able to REFUSE a malformed world at construction and name the field.
[[nodiscard]] inline std::string world_shape_refusal(const WorldShape& world) {
    if (world.world_size == 0) { return "world_size must be at least 1"; }
    if (world.rank >= world.world_size) {
        return "rank " + std::to_string(world.rank) + " is outside world_size " +
               std::to_string(world.world_size);
    }
    if (world.axis == ParallelAxis::None && world.world_size != 1) {
        return "world_size " + std::to_string(world.world_size) +
               " with axis=none: a multi-device world must name the axis it splits on, "
               "because the KV granularity contract depends on which one it is";
    }
    if (world.axis != ParallelAxis::None && world.world_size == 1) {
        return "axis=" + std::string(axis_name(world.axis)) +
               " with world_size 1: a single-device world is axis=none";
    }
    return {};
}

[[nodiscard]] inline bool world_shape_valid(const WorldShape& world) {
    return world_shape_refusal(world).empty();
}

// ---------------------------------------------------------------------------
// 2. Geometry in, shard ranges out
// ---------------------------------------------------------------------------

struct ModelGeometry {
    std::uint32_t text_layers = 0;
    std::uint32_t q_heads     = 0;
    std::uint32_t kv_heads    = 0;
    std::uint32_t head_dim    = 0;
    // Output columns of the sharded weights (the N of the linear layers that are split on
    // TP). For a GQA model the attention QKV projection's N is
    // (q_heads + 2 * kv_heads) * head_dim, which is NOT kv_heads*head_dim -- the caller
    // supplies the real number so this header does not have to guess a layout.
    std::uint32_t weight_columns = 0;
};

struct Range {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;

    [[nodiscard]] std::uint32_t count() const noexcept { return end - begin; }
    [[nodiscard]] bool empty() const noexcept { return end <= begin; }
    friend bool operator==(const Range&, const Range&) = default;
};

struct ShardPlan {
    bool ok = false;
    std::string reason;  // non-empty iff !ok

    ParallelAxis axis = ParallelAxis::None;
    Range weight_columns;   // TP: this rank's N block. PP: the whole thing on every stage.
    Range kv_heads;         // TP: this rank's KV heads. PP: all of them on every stage.
    Range text_layers;      // PP: this stage's layers. TP: all of them on every rank.

    std::uint32_t q_per_kv = 0; // unchanged by both axes; the GQA group size

    // The two facts the KV policy has to consult. See section 4.
    [[nodiscard]] bool page_is_layer_complete() const noexcept {
        return axis != ParallelAxis::Pipeline;
    }
    [[nodiscard]] bool residency_decision_is_replicated() const noexcept {
        // "Replicated" = the admit/refuse verdict is a single value for the world, i.e. no
        // rank may disagree. TRUE for the single device and for the pipeline axis (where
        // the verdict is per STAGE and each stage's own verdict is complete for the pages
        // that stage reads), FALSE for the tensor axis unless the caller adds the AND of
        // section 5.
        return axis != ParallelAxis::Tensor;
    }
};

[[nodiscard]] inline ShardPlan refuse(std::string reason) {
    ShardPlan plan;
    plan.ok     = false;
    plan.reason = std::move(reason);
    return plan;
}

[[nodiscard]] inline ShardPlan plan_shards(const WorldShape& world,
                                           const ModelGeometry& geometry) {
    const std::string world_refusal = world_shape_refusal(world);
    if (!world_refusal.empty()) { return refuse(world_refusal); }
    if (geometry.text_layers == 0 || geometry.q_heads == 0 || geometry.kv_heads == 0 ||
        geometry.head_dim == 0) {
        return refuse("model geometry is incomplete (text_layers / q_heads / kv_heads / "
                      "head_dim must all be positive)");
    }
    if (geometry.q_heads % geometry.kv_heads != 0) {
        return refuse("q_heads " + std::to_string(geometry.q_heads) +
                      " is not a multiple of kv_heads " + std::to_string(geometry.kv_heads) +
                      ": the GQA group size is not an integer, so no head split is defined");
    }

    ShardPlan plan;
    plan.axis           = world.axis;
    plan.q_per_kv       = geometry.q_heads / geometry.kv_heads;
    plan.weight_columns = Range{0, geometry.weight_columns};
    plan.kv_heads       = Range{0, geometry.kv_heads};
    plan.text_layers    = Range{0, geometry.text_layers};

    switch (world.axis) {
    case ParallelAxis::None:
        plan.ok = true;
        plan.reason.clear();
        return plan;

    case ParallelAxis::Tensor: {
        // KV HEADS: the split is only legal when it keeps every GQA group whole. Rank r
        // owns q_heads/world_size consecutive query heads and must own exactly the kv heads
        // those queries attend to; if kv_heads does not divide, a group straddles two ranks
        // and a rank's queries need a kv head it does not hold -- which no amount of
        // cross-rank reduction can repair, because the missing head's K/V bytes are simply
        // not on that card.
        if (geometry.kv_heads % world.world_size != 0) {
            return refuse("kv_heads " + std::to_string(geometry.kv_heads) +
                          " is not divisible by tp world_size " +
                          std::to_string(world.world_size) +
                          ": the split would cut a GQA group across ranks, so a rank's "
                          "query heads would attend to kv heads it does not own. Choose a "
                          "world_size that divides kv_heads, or replicate the KV heads "
                          "instead of splitting them (which is not tensor parallelism).");
        }
        // WEIGHT N: a column block must be whole, because a rank owns those output
        // elements and the all-reduce sums PARTIAL SUMS, not partial outputs.
        if (geometry.weight_columns % world.world_size != 0) {
            return refuse("weight N " + std::to_string(geometry.weight_columns) +
                          " is not divisible by tp world_size " +
                          std::to_string(world.world_size) +
                          ": a partial column block would put one output element's partial "
                          "sum on two ranks with no way to distinguish it from a complete "
                          "one. Pad N, or pick a world_size that divides it.");
        }
        const std::uint32_t per_rank_heads  = geometry.kv_heads / world.world_size;
        const std::uint32_t per_rank_cols   = geometry.weight_columns / world.world_size;
        const std::uint32_t per_rank_layers = geometry.text_layers; // layers stay whole
        plan.kv_heads       = Range{world.rank * per_rank_heads, (world.rank + 1) * per_rank_heads};
        plan.weight_columns = Range{world.rank * per_rank_cols, (world.rank + 1) * per_rank_cols};
        plan.text_layers    = Range{0, per_rank_layers};
        plan.ok             = true;
        plan.reason.clear();
        return plan;
    }

    case ParallelAxis::Pipeline: {
        // LAYERS: a balanced split, and it is ALLOWED to be uneven. Unlike the head split,
        // an uneven pipeline costs only balance -- each stage's layers are complete within
        // that stage, so no single layer is ever cut. The first `remainder` stages take one
        // extra layer, which is the usual convention and is deterministic.
        const std::uint32_t base      = geometry.text_layers / world.world_size;
        const std::uint32_t remainder = geometry.text_layers % world.world_size;
        if (base == 0) {
            return refuse("text_layers " + std::to_string(geometry.text_layers) +
                          " is smaller than pp world_size " +
                          std::to_string(world.world_size) +
                          ": a stage would own no layer at all. Use fewer stages than "
                          "layers, or split on heads instead.");
        }
        const std::uint32_t begin = world.rank * base + (world.rank < remainder ? world.rank
                                                                               : remainder);
        const std::uint32_t end   = begin + base + (world.rank < remainder ? 1U : 0U);
        plan.text_layers    = Range{begin, end};
        plan.kv_heads       = Range{0, geometry.kv_heads};       // all, on every stage
        plan.weight_columns = Range{0, geometry.weight_columns}; // all, on every stage
        plan.ok             = true;
        plan.reason.clear();
        return plan;
    }
    }
    return refuse("unhandled parallel axis");
}

// ---------------------------------------------------------------------------
// 3. Per-rank persistence naming
// ---------------------------------------------------------------------------
//
// Today the cold artifacts are named for the ARTIFACT alone -- a single file per artifact:
//   <artifact>.kvrowscale.bin            the table (src/product/kv_rowscale_persist.h)
//   <artifact>.kvrowscale.bin.d          capture records
//   <artifact>.kvrowscale.bin.skip
//   <artifact>.kvrowscale.bin.rejected
//   <artifact>.kvrowscale.bin.words.bin
// The suffix is `kKvRowScaleTableSuffix` = ".kvrowscale.bin". (The KV row-scale table is
// the persisted artifact that carries the per-layer/per-head KV write scales; the cold
// spill file is named separately in program_impl.h and gets a `file_slot` per page.)
//
// Under a split that changes the KV's own shape (kv heads per rank, or layers per stage)
// the table is no longer the artifact's table -- it is the artifact's table FOR ONE RANK.
// Two ranks writing the same name would silently overwrite each other with tables of a
// different geometry, and a reader started with a different world would then apply a table
// whose row count does not match its own KV. So the name carries the shard, and the reader
// REFUSES a name whose shard is not its own -- which is the same shape of guard the recall
// journal already uses for its stride ("the stride the writer used, so a reader with a
// different geometry refuses instead of mis-reading", turn_recall_journal.h).

inline constexpr std::string_view kShardedNameSeparator = ".";

// Inserts the shard tag immediately BEFORE the suffix:
//   base="m.kvrowscale.bin", {tp, rank 1, world 4} -> "m.tp1of4.kvrowscale.bin"
// A single-device world returns `base` unchanged, so the existing artifact names and every
// existing file on disk keep working.
[[nodiscard]] inline std::string sharded_artifact_name(std::string_view base,
                                                       const WorldShape& world,
                                                       std::string_view suffix) {
    if (world.axis == ParallelAxis::None) { return std::string(base); }
    const std::string tag = std::string(axis_name(world.axis)) + std::to_string(world.rank) +
                            "of" + std::to_string(world.world_size);
    if (base.size() >= suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return std::string(base.substr(0, base.size() - suffix.size())) + "." + tag +
               std::string(suffix);
    }
    // No recognised suffix: append rather than silently drop part of the name.
    return std::string(base) + "." + tag;
}

struct ShardNameParse {
    bool ok          = false;
    bool sharded     = false;
    ParallelAxis axis = ParallelAxis::None;
    std::uint32_t rank       = 0;
    std::uint32_t world_size = 0;
    std::string reason;
};

// Recognises the tag produced above. An unsharded name parses as sharded=false, which the
// acceptance rule below treats as a REFUSAL for a multi-device world -- an unsharded table
// was written by a single-device run and its geometry does not describe one rank's KV.
[[nodiscard]] inline ShardNameParse parse_sharded_artifact_name(std::string_view name) {
    ShardNameParse out;
    const std::size_t tag_at = name.find(".tp");
    const std::size_t pp_at  = name.find(".pp");
    std::size_t at           = std::string_view::npos;
    ParallelAxis axis        = ParallelAxis::None;
    if (tag_at != std::string_view::npos && (pp_at == std::string_view::npos || tag_at < pp_at)) {
        at   = tag_at;
        axis = ParallelAxis::Tensor;
    } else if (pp_at != std::string_view::npos) {
        at   = pp_at;
        axis = ParallelAxis::Pipeline;
    }
    if (at == std::string_view::npos) {
        out.ok      = true;
        out.sharded = false;
        out.reason  = "name carries no shard tag: it was written by a single-device run";
        return out;
    }
    const std::size_t digits_at = at + kShardedNameSeparator.size() + axis_name(axis).size();
    const std::size_t of_at     = name.find("of", digits_at);
    if (of_at == std::string_view::npos || of_at == digits_at) {
        out.reason = "shard tag is malformed: expected .<axis><rank>of<world>";
        return out;
    }
    // Stop the world size at the next separator so a suffix such as .kvrowscale.bin does
    // not end up inside the number.
    std::size_t end = name.size();
    for (std::size_t i = of_at + 2; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') { end = i; break; }
    }
    if (end == of_at + 2) {
        out.reason = "shard tag is malformed: world size is missing";
        return out;
    }
    auto to_u32 = [](std::string_view text, bool& good) {
        std::uint32_t value = 0;
        for (const char c : text) {
            if (c < '0' || c > '9') { good = false; return 0U; }
            value = value * 10U + static_cast<std::uint32_t>(c - '0');
        }
        return value;
    };
    bool rank_good = true, world_good = true;
    out.rank       = to_u32(name.substr(digits_at, of_at - digits_at), rank_good);
    out.world_size = to_u32(name.substr(of_at + 2, end - of_at - 2), world_good);
    if (!rank_good || !world_good || out.world_size == 0 || out.rank >= out.world_size) {
        out.reason = "shard tag is malformed or out of range";
        return out;
    }
    out.ok      = true;
    out.sharded = true;
    out.axis    = axis;
    out.reason.clear();
    return out;
}

// The reader's rule. Returns empty on acceptance, a reason otherwise. This is the whole
// point of the naming: a table whose shard is not the running shard must not be applied,
// because its row count describes a different KV geometry.
[[nodiscard]] inline std::string sharded_name_acceptance(std::string_view name,
                                                         const WorldShape& world) {
    const ShardNameParse parsed = parse_sharded_artifact_name(name);
    if (!parsed.ok) { return "cannot parse the persisted name '" + std::string(name) + "': " + parsed.reason; }
    if (world.axis == ParallelAxis::None) {
        if (parsed.sharded) {
            return "persisted name '" + std::string(name) +
                   "' carries a shard tag but this run is single-device: this is the wrong "
                   "kind of table entirely";
        }
        return {};
    }
    if (!parsed.sharded) {
        return "persisted name '" + std::string(name) +
               "' carries no shard tag but this run has world_size " +
               std::to_string(world.world_size) + " on axis " +
               std::string(axis_name(world.axis)) +
               ": an unsharded table describes the whole KV, not this rank's shard";
    }
    if (parsed.axis != world.axis) {
        return "persisted name '" + std::string(name) + "' was written on axis " +
               std::string(axis_name(parsed.axis)) + " but this run is on axis " +
               std::string(axis_name(world.axis));
    }
    if (parsed.rank != world.rank || parsed.world_size != world.world_size) {
        return "persisted name '" + std::string(name) + "' is for " +
               std::string(axis_name(parsed.axis)) + std::to_string(parsed.rank) + "of" +
               std::to_string(parsed.world_size) + " but this run is " +
               std::string(axis_name(world.axis)) + std::to_string(world.rank) + "of" +
               std::to_string(world.world_size) +
               ": a table from a different shard cannot be applied";
    }
    return {};
}

// ---------------------------------------------------------------------------
// 4. The page residency bit across a partition  <-- the load-bearing part
// ---------------------------------------------------------------------------
//
// The KV policy's eviction unit is ONE PAGE, and "one page is one residency bit"
// (cold_host_tier.h). Splitting the KV across cards splits the STATE that carries that bit
// while leaving its DECISION INPUTS shared, and that asymmetry is what section 5 measures.
//
// Inputs of the decision, all shared by construction:
//   * `frontier`  the committed-token count;
//   * `layer_windows`  the per-layer sliding windows (a config, replicated);
//   * the cold byte budgets, which are config and therefore nominally identical.
// State of the decision, all PER RANK once the world is split:
//   * `ColdHostTier::usage_` / `ColdTierUsage`, the counter `admit_cold_page` reads;
//   * the `HostKVArena` (its capacity is a per-rank byte cap);
//   * `allocate_cold_disk_file_slot()`, whose free list is process-global, i.e. per rank.
// So two ranks can run the SAME predicate to the SAME answer and still reach DIFFERENT
// admits, purely because one rank's arena is full. That is not a bug in the predicate, and
// no amount of making the predicate smarter fixes it.

enum class ResidencyState : std::uint8_t {
    FullyAbsent,       // every rank refuses the page
    FullyResident,     // every rank admits it
    PartiallyResident, // ranks disagree -- A STATE THE MODEL DOES NOT DEFINE
};

[[nodiscard]] inline std::string_view residency_state_name(ResidencyState state) noexcept {
    switch (state) {
    case ResidencyState::FullyAbsent: return "fully-absent";
    case ResidencyState::FullyResident: return "fully-resident";
    case ResidencyState::PartiallyResident: return "partially-resident";
    }
    return "unknown";
}

// The one bit, as the tree means it. `per_rank_admit` must have world_size entries.
[[nodiscard]] inline ResidencyState residency_state(const std::vector<std::uint8_t>& per_rank_admit) {
    if (per_rank_admit.empty()) { return ResidencyState::FullyAbsent; }
    const bool any = per_rank_admit.front() != 0;
    for (const std::uint8_t admit : per_rank_admit) {
        if ((admit != 0) != any) { return ResidencyState::PartiallyResident; }
    }
    return any ? ResidencyState::FullyResident : ResidencyState::FullyAbsent;
}

// The replicated decision: the page leaves the device only if EVERY rank can take it. This
// is an AND-reduce over ranks, and under the tensor axis it is a NEW collective on the
// eviction driver's critical path -- the price of keeping the page atomic.
[[nodiscard]] inline bool global_admit(const std::vector<std::uint8_t>& per_rank_admit) {
    for (const std::uint8_t admit : per_rank_admit) {
        if (admit == 0) { return false; }
    }
    return true;
}

// THE FACTORING QUESTION, stated as the property it is.
//
// `cold_host_page_is_read_free(page, page_tokens, frontier, layer_windows)` is a pure
// CONJUNCTION over layers: it returns false if ANY layer has window 0 or has not moved
// past the page, and true otherwise. A conjunction factors across a partition of its
// conjuncts: AND over all layers == AND over groups of (AND over the group). So a LAYER
// partition (the pipeline axis) can evaluate the predicate per stage and AND the answers
// and get EXACTLY the global answer -- the replication is free and exact.
//
// It does NOT factor across a partition of the KV's CONTENT (the tensor axis). Heads are
// not conjuncts of this predicate at all: the predicate is a statement about LAYERS. And
// the KV content of head h is not a function of the content of head h', so "this rank can
// drop its heads for page p" says nothing about whether page p is still a legal attention
// input -- see residency_factors_over() and the counterexample in
// tests/test_shard_plan.cpp.
[[nodiscard]] inline bool read_free_predicate_factors_over(ParallelAxis axis) noexcept {
    return axis != ParallelAxis::Tensor;
}

// Whether "the page is resident" can remain ONE bit with no new cross-rank agreement, i.e.
// whether a per-rank decision is already a world decision.
[[nodiscard]] inline bool residency_decision_needs_replication(ParallelAxis axis) noexcept {
    return axis == ParallelAxis::Tensor;
}

// Whether the page's content is materially complete on one card, which is what
// cold_host_tier.h's GRANULARITY and the single-cold-slot-base sentinel both require.
//   pipeline -> FALSE. The page's text layers live on different stages, so "one logical
//               page ACROSS ALL TEXT LAYERS" cannot be one resident object anywhere, and
//               the sentinel's ONE slot base indexed "in every layer's cold_slots at once"
//               no longer has a single layer set to index.
//   tensor   -> TRUE. Every rank holds every layer for its own heads, so each rank's page
//               is layer-complete; what it is not is head-complete, which is why the
//               tensor axis costs the RESIDENCY DECISION and not the page LAYOUT.
[[nodiscard]] inline bool page_content_is_layer_complete(ParallelAxis axis) noexcept {
    return axis != ParallelAxis::Pipeline;
}

// Whether the recall journal's single-stride-per-page rule survives the axis. The journal
// stores ONE `layer_bytes` ("per-layer slot stride as written") per page and refuses
// RecallCodec::Mixed at compile time. A head split multiplies every layer's stride by the
// same 1/world_size -- still ONE stride, still one codec per layer -- so the record shape
// survives untouched. A layer split does not: stages may legitimately choose DIFFERENT
// per-layer codecs (that is exactly what `--kv-layer-storage 0-7:rk4v4,4-7:int8` does), which
// makes RecallCodec::Mixed REACHABLE and the existing static_assert a wall the feature
// cannot get past.
[[nodiscard]] inline bool recall_record_shape_survives(ParallelAxis axis) noexcept {
    return axis != ParallelAxis::Pipeline;
}

// The per-layer spill cell path, which is the DISK tier's region name and therefore the
// thing a second rank overwrites. The tree builds it today as (src/targets/qwen3_6/impl/
// runtime/program_impl.h, the Disk / HostThenDisk arm):
//
//     const std::string dir = cold_disk_path.empty() ? std::string("/tmp") : cold_disk_path;
//     const std::string path = dir + "/ninfer_cold_L" + std::to_string(layer) + ".slot";
//
// -- a LAYER axis with NO rank axis, and `dir` defaults to "/tmp". Two ranks of the same
// world therefore open the same file, size the same `file_slot` space from it, and both
// start at `(L0.slot, offset 0)`: the per-process slot allocator
// (`allocate_cold_disk_file_slot`) hands out the same first slot in every process, so the
// writes do not merely interleave, they COLLIDE deterministically. Spilling is not
// best-effort: a collision silently swaps two different pages' bytes.
//
// This function is the rank-aware spelling. For a single-device world it returns the
// string above BYTE FOR BYTE, so every name on disk today keeps working and the change is
// behaviour-preserving by construction.
[[nodiscard]] inline std::string cold_spill_cell_path(std::string_view dir,
                                                      std::uint32_t layer,
                                                      const WorldShape& world) {
    std::string out(dir);
    out += "/ninfer_cold_L";
    out += std::to_string(layer);
    if (world.axis != ParallelAxis::None) {
        out += ".r";
        out += std::to_string(world.rank);
        out += "of";
        out += std::to_string(world.world_size);
    }
    out += ".slot";
    return out;
}

// ---------------------------------------------------------------------------
// 5. WHAT IS MISSING (named, not stubbed)
// ---------------------------------------------------------------------------
//
// Everything above is arithmetic and naming and is exercised on a host. The following are
// REQUIRED to make any of it run, and none of them exists:
//
//   * a rank/world plumbing layer: process launch, rank discovery, and the deterministic
//     iteration order the replicated admit decision needs (all ranks must walk the same
//     page order or "replicated" is meaningless);
//   * the collectives themselves -- for tensor parallelism an all-reduce over the attention
//     combine triple (row max, row sumexp, partial output) per layer, and an all-reduce AND
//     of the eviction admit verdict;
//   * `cudaDeviceEnablePeerAccess` / `cudaMemcpyPeer` (or an NCCL equivalent) plus a
//     topology decision, since the engine has neither today;
//   * a per-rank weight loader: the artifact reader produces ONE byte stream and there is
//     no N-dim column slicer anywhere in src/artifact/;
//   * a rank/world field in the recall journal record. RecallRecord is exactly 64 bytes
//     with two free words (`pad0`, `pad1`) and a static_assert pinning the size, so the
//     field can be added without moving the stride -- but it has not been;
//   * a device count > 1 exercised anywhere. tools/archkit/_GPU_MATRIX.md's "我们的落地
//     路线" item 1 ("单进程 TP 骨架") is a plan, and the bounded probe in this file's
//     header comment says none of it is code.

// ---------------------------------------------------------------------------
// 6. THE ENGINE'S OWN SHARD CONTRACT -- is there anything in this tree that can CARRY a
//    world, and the refusal that names the pieces when there is not
// ---------------------------------------------------------------------------
//
// WHY THIS SECTION EXISTS, AND WHAT IT IS NOT.
//
// Sections 1-4 are ARITHMETIC. Given a world and a geometry they answer "which slice does
// this rank own", and they answer yes for world_size 4 on four cards. That is correct: the
// arithmetic IS defined there and the slices ARE unambiguous. Nothing in it is a claim that
// the ENGINE can carry such a world.
//
// This section is the missing half, and it asks the ENGINE's question, which is a different
// one: is there anything in this tree that can BIND a second device, MOVE a byte between two
// devices, or READ a per-rank weight stream? All of it is answered false in this revision, by
// measurement rather than by opinion, and this section turns that from a SILENCE into a
// NAMED REFUSAL.
//
// WHAT IT REPLACES, with the file:line that makes it a measurement rather than a story.
// Before this section a machine with two cards got a SILENT SINGLE-CARD run:
//   * the engine's one construction site is `DeviceContext device;`
//     (src/runtime/engine/engine.cpp:401), whose default argument is `device_id = 0`
//     (src/core/device.h:42);
//   * `cudaGetDeviceCount` is consulted at src/core/device.cu:50 ONLY to range-check that 0
//     (`if (device_id < 0 || device_id >= count)`, :55);
//   * there is NO --tp / --devices / --gpu flag anywhere under apps/ (measured: zero hits),
//     so an operator cannot pass a second device id even if one existed;
//   * src/core/device_sm_count.h:42 asks `cudaGetDevice` for the CURRENT device, i.e. it
//     reports whichever one DeviceContext bound -- device 0.
// Device 1 is never named, never bound and never missed. The ONE sentence in the tree that
// mentions the physical device count is src/core/virtual_device.h:402
// (`physical_device_count != 1`), and it refuses the SIMULATION with "Use them" -- an
// instruction this revision cannot follow, because the only type that can bind a second
// device is `ExecutionContext` (src/core/device.h:94-102) and its ENGINE callers number ZERO
// (measured: its two construction sites in the whole tree are
// tests/ops/test_allreduce.cpp:394 and
// tests/targets/qwen3_6_27b/test_sharded_materialization.cpp:390).
//
// ONE DELIBERATE NON-INCLUDE, so nobody "fixes" it later. The virtual-device key names below
// are spelled as LITERALS and not as core/virtual_device.h's `kVirtualDevicesEnv` constants,
// because that header INCLUDES THIS ONE (its line 61) and the include would be circular. The
// two spellings are pinned against each other by the test that ships with this section
// (tests/test_shard_contract.cpp G5), so a rename in either file goes red there rather than
// producing a refusal that names a key nobody sets.
//
// WHY THE REFUSAL IS A FUNCTION OF THE FACTS AND NOT A CONSTANT.
// A hardcoded "no multi-device" would be the number-based gate this project keeps deleting
// (tools/archkit/_GPU_MATRIX.md: "不是靠数字猜"; tests/test_multidev_wiring.cpp W1's own
// invariant). So every fact below is a BOOLEAN PARAMETER, the refusal is COMPUTED from them,
// and with all four set the SAME call returns empty. The test walks all sixteen subsets, so
// the refusal cannot be satisfied by a constant that ignores its input.

// The four things a world spanning MORE THAN ONE PHYSICAL DEVICE needs, each a measurement of
// this revision rather than a plan. Set a field to true only in the commit that lands the
// thing behind it, together with the evidence that it is in the BUILD (not merely on disk).
struct CrossDeviceFacts {
    // THE TRANSPORT, in TWO pieces, and NEITHER is the engine's.
    //   * src/ops/common/allreduce.cu (31,125 B) is an AMD HIP port: `#include
    //     <hip/hip_runtime.h>` at :39, `__hip_atomic_store` / `__hip_atomic_load` at
    //     :179-181, `hipStreamIsCapturing` at :279, `hipExtMallocWithFlags` at :321. Its own
    //     PROVENANCE header (:9) says "ADDITIVE, NOT wired into any build target
    //     (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds
    //     it deliberately". It has ZERO hits in src/CMakeLists.txt, AND it cannot be compiled
    //     by the CUDA toolchain this box has: MEASURED, `nvcc -std=c++20 -c -arch=sm_120
    //     src/ops/common/allreduce.cu` dies at :39 with
    //     "hip/hip_runtime.h: No such file or directory", and no hip_runtime.h exists anywhere
    //     on this box. Transport on disk is not transport in the build, and this one is not
    //     even transport this toolchain can build.
    //   * tools/mg4/tp2_sum_stress.cu (87,841 B) IS CUDA and IS BUILT
    //     (build/tests/ninfer_tp2_sum_stress, registered at tests/CMakeLists.txt:842). It is
    //     DELIBERATELY not a ctest test -- its own registration comment: it takes the GPU, and
    //     a ctest test that grabbed the device would contend with whichever model chain holds
    //     it -- and it MEASURES the transport's cost rather than being a transport the engine
    //     links. So the one CUDA-side piece that exists is evidence ABOUT the transport.
    bool collective_transport_in_build = false;
    // A rank/world plumbing layer: process launch, rank discovery, and the deterministic
    // iteration order the replicated admit decision needs (sections 4 and 5 state why). The
    // environment surface for a PHYSICAL world does not exist: `NINFER_WORLD_SIZE` has 0 hits
    // in the tree and `NINFER_RANK` has 1, and that one is `NINFER_RANKING_PATH`
    // (src/targets/qwen3_8_flash_next/impl/load/materialized.cpp:31) -- a ranking path, not a
    // rank. The only rank read in the tree is `NINFER_VIRTUAL_RANK`
    // (src/core/virtual_device.h:144), which resolves a rank inside ONE PROCESS ON ONE
    // PHYSICAL CARD.
    bool rank_world_plumbing = false;
    // A per-rank weight loader. The artifact reader produces ONE byte stream (section 5 names
    // this), and the N-dim column slicer that would cut it per rank (src/core/n_dim_slice.h)
    // is a HOST-ONLY header with no loader behind it. The register is still slotted.
    bool per_rank_weight_loader = false;
    // A cross-device capture. src/core/decode_graph.h captures ONE device, and the donor's
    // measured constraint is that two LIVE captures fail with `cudaErrorStreamCaptureMerge`
    // (docs/gfx906/TP2-SLICES.md, quoted at src/core/virtual_device.h:120-123: one fork/join
    // graph spanning both devices, +15% nodes, 1888 at tp2 against 640 at tp1). The plan for
    // it is src/core/decode_graph_peer.h; the capture is not there.
    bool cross_device_decode_graph = false;
};

// THE FACTS OF THIS REVISION, spelled once so no call site can disagree with another. Every
// field is false and every false is a measurement with the file:line above behind it.
inline constexpr CrossDeviceFacts kCrossDeviceFactsThisRevision{};

// The pieces, by name, in the field order above, so a refusal can LIST them and a test can
// count them without a second spelling of the list.
inline constexpr std::string_view kCrossDevicePieceNames[] = {
    "a collective transport in the BUILD (the tree's only cross-rank allreduce is an AMD HIP "
    "port at src/ops/common/allreduce.cu:39 -- not in the link and not compilable by nvcc)",
    "a rank/world plumbing layer (process launch, rank discovery, deterministic page order)",
    "a per-rank weight loader (no N-dim column slicer behind src/artifact/)",
    "a cross-device decode graph (src/core/decode_graph.h captures one device)",
};
inline constexpr std::size_t kCrossDevicePieceCount = 4;

// True iff EVERY piece is present, i.e. iff this build can carry a world across more than one
// physical device. A conjunction, and deliberately not a threshold: one missing piece is a
// world that cannot run.
[[nodiscard]] inline bool cross_device_world_possible(const CrossDeviceFacts& facts) noexcept {
    return facts.collective_transport_in_build && facts.rank_world_plumbing &&
           facts.per_rank_weight_loader && facts.cross_device_decode_graph;
}

// The pieces this build does NOT have, in kCrossDevicePieceNames order. Empty iff
// cross_device_world_possible(). A vector of strings rather than a count, because a refusal
// that says "4 pieces are missing" makes the reader go and find out WHICH.
[[nodiscard]] inline std::vector<std::string> missing_cross_device_pieces(
    const CrossDeviceFacts& facts) {
    std::vector<std::string> missing;
    if (!facts.collective_transport_in_build) { missing.emplace_back(kCrossDevicePieceNames[0]); }
    if (!facts.rank_world_plumbing) { missing.emplace_back(kCrossDevicePieceNames[1]); }
    if (!facts.per_rank_weight_loader) { missing.emplace_back(kCrossDevicePieceNames[2]); }
    if (!facts.cross_device_decode_graph) { missing.emplace_back(kCrossDevicePieceNames[3]); }
    return missing;
}

// ---------------------------------------------------------------------------
// The world request, as read from the environment
// ---------------------------------------------------------------------------
// Named once so nothing can spell it differently. These are the keys a LAUNCHER would set for
// a physical world, and they are deliberately NOT the virtual-device keys in
// core/virtual_device.h: `NINFER_VIRTUAL_DEVICES=N` means N LOGICAL RANKS ON ONE CARD (a
// capacity simulation), while `NINFER_WORLD_SIZE=N` means N RANKS THAT WANT N CARDS.
// Collapsing the two spellings would be the class of mistake this file's world_shape_refusal
// already refuses for axes.
inline constexpr std::string_view kWorldSizeEnv = "NINFER_WORLD_SIZE";
inline constexpr std::string_view kRankEnv      = "NINFER_RANK";

struct WorldRequest {
    bool requested           = false; // NINFER_WORLD_SIZE was set to something
    std::uint32_t world_size = 1;
    std::uint32_t rank       = 0;
    std::string world_size_raw;
    std::string rank_raw;
};

// Strict, and strict on purpose: a typo must be a refusal rather than a quiet default. Same
// shape as core/virtual_device.h's own parse (its comment: "a typo is a refusal rather than
// rank 0").
[[nodiscard]] inline std::uint32_t parse_world_u32(std::string_view text, bool& good) noexcept {
    std::uint32_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') { good = false; return 0U; }
        value = value * 10U + static_cast<std::uint32_t>(c - '0');
        if (value > 4096U) { good = false; return 0U; }
    }
    return value;
}

[[nodiscard]] inline std::string read_world_env(std::string_view name) {
    const std::string key(name);
    const char* value = std::getenv(key.c_str());
    return value == nullptr ? std::string{} : std::string(value);
}

// The reader. Kept separate from the refusal so the parse is testable without an environment
// -- the same split core/virtual_device.h uses, and for the same reason.
[[nodiscard]] inline WorldRequest world_request_from_environment() {
    WorldRequest request;
    request.world_size_raw = read_world_env(kWorldSizeEnv);
    request.rank_raw       = read_world_env(kRankEnv);
    request.requested      = !request.world_size_raw.empty();
    if (!request.requested) { return request; }
    bool size_good = true;
    bool rank_good = true;
    request.world_size = parse_world_u32(request.world_size_raw, size_good);
    request.rank       = request.rank_raw.empty()
                             ? 0U
                             : parse_world_u32(request.rank_raw, rank_good);
    if (!size_good) { request.world_size = 0U; }
    if (!rank_good) { request.rank = 0U; }
    return request;
}

// ---------------------------------------------------------------------------
// The refusal
// ---------------------------------------------------------------------------
// Empty iff this build can honour the declared world. Every branch names the field or the
// piece it failed on, because a refusal that does not say which line is missing costs the
// reader the time it was meant to save.
[[nodiscard]] inline std::string cross_device_world_refusal(const WorldRequest& request,
                                                            const CrossDeviceFacts& facts,
                                                            std::uint32_t physical_device_count) {
    // Nothing was asked: the guard is OFF and the run is byte-for-byte what it was before this
    // section existed. This is the arm that keeps the whole contract behaviour-preserving.
    if (!request.requested) { return {}; }

    if (request.world_size == 0U) {
        std::string out = std::string(kWorldSizeEnv) + "=\"";
        out += request.world_size_raw;
        out += "\" is not a rank count: it must be a positive integer. A value that cannot be "
               "read is refused rather than defaulted, because a run that quietly became "
               "single-device is indistinguishable from a successful world at the point where "
               "the numbers are read.";
        return out;
    }
    // A single-rank world is the real device. There is nothing to span, nothing to collect and
    // nothing to refuse -- and saying so is what makes the multi-rank branch below mean
    // something.
    if (request.world_size == 1U) {
        if (request.rank != 0U) {
            std::string out = std::string(kRankEnv) + "=";
            out += std::to_string(request.rank);
            out += " with world_size 1: rank 0 is the only rank of a one-rank world";
            return out;
        }
        return {};
    }
    if (request.rank >= request.world_size) {
        std::string out = std::string(kRankEnv) + "=";
        out += std::to_string(request.rank);
        out += " is outside world_size ";
        out += std::to_string(request.world_size);
        out += ": a rank must be one of the world's own (see world_shape_refusal above, which "
               "states the same rule for a WorldShape)";
        return out;
    }
    // THE HARDWARE ITSELF. A world of N ranks needs N cards before anything else is true, and
    // this is a fact about the machine rather than about the tree.
    if (request.world_size > physical_device_count) {
        std::string out = "a world of ";
        out += std::to_string(request.world_size);
        out += " ranks is declared (";
        out += kWorldSizeEnv;
        out += "=";
        out += request.world_size_raw;
        out += ") but this machine has ";
        out += std::to_string(physical_device_count);
        out += " CUDA device(s). The shard arithmetic in sections 1-4 is defined for any "
               "world_size; a world of N ranks has nowhere to run unless N devices exist. Use "
               "a world_size no larger than the device count, or split on a different axis.";
        return out;
    }
    // THE PIECES. All the cards exist and the world still cannot run, because the tree has
    // nothing that binds a second one. This is the branch the silent single-card run used to
    // hide in.
    if (!cross_device_world_possible(facts)) {
        const std::vector<std::string> missing = missing_cross_device_pieces(facts);
        std::string out = "a world of ";
        out += std::to_string(request.world_size);
        out += " ranks is declared (";
        out += kWorldSizeEnv;
        out += "=";
        out += request.world_size_raw;
        out += ") and this machine has ";
        out += std::to_string(physical_device_count);
        out += " CUDA devices, but this build cannot span them. MISSING, by name:";
        for (const std::string& piece : missing) {
            out += " (";
            out += piece;
            out += ")";
        }
        out += ". What this build CAN do: world_size 1 on any single card, and a world of N "
               "LOGICAL ranks on ONE physical card through NINFER_VIRTUAL_DEVICES together "
               "with its acknowledgement key -- a capacity simulation of the shard shape, "
               "never a two-card measurement. Unset ";
        out += kWorldSizeEnv;
        out += " to run single-card, or set the virtual pair for the single-card simulation.";
        return out;
    }
    // Every piece is present: the world is honourable and this function must say so. A
    // hardcoded refusal cannot reach this line, which is what the sixteen-subset test walks.
    return {};
}

// ---------------------------------------------------------------------------
// The other half: a machine with cards this run will not use must SAY SO
// ---------------------------------------------------------------------------
// A single-card run on a multi-card box is a legitimate run, so this is a notice and not a
// refusal. What is not legitimate is not knowing: before this, device 0 was bound and the
// other cards were idle and unmentioned (the file:line at the top of this section). Empty
// whenever there is nothing to say -- one card, or a world that already explains the ranks.
[[nodiscard]] inline std::string idle_device_notice(std::uint32_t physical_device_count,
                                                    const WorldShape& world) {
    if (physical_device_count <= 1U) { return {}; }
    if (world.world_size > 1U) { return {}; }
    std::string out = "ninfer: this machine has ";
    out += std::to_string(physical_device_count);
    out += " CUDA devices and this run will use exactly ONE (device 0). The other ";
    out += std::to_string(physical_device_count - 1U);
    out += " are idle.\n";
    out += "  Why: this build has no flag that binds a second device (measured: no --tp / "
           "--devices / --gpu anywhere under apps/), and no cross-device transport in the "
           "link (the tree's only cross-rank allreduce is an AMD HIP port, "
           "src/ops/common/allreduce.cu:39).\n";
    out += "  Declaring a world (";
    out += kWorldSizeEnv;
    out += "=N) is REFUSED BY NAME in this revision, and the refusal lists the missing "
           "pieces.\n";
    out += "  For the single-card CAPACITY simulation of the shard shape, set "
           "NINFER_VIRTUAL_DEVICES=N with NINFER_VIRTUAL_DEVICES_ACK=yes "
           "(core/virtual_device.h).\n";
    out += "  A number from a single-card run is NOT a scaling result.\n";
    return out;
}

} // namespace ninfer::multi
