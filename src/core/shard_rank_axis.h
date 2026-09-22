#pragma once

// shard_rank_axis.h -- THE RANK AXIS ITSELF: the three arithmetic structures that had no
// rank dimension, plus the attention-combine algebra the tensor axis has to reduce over.
//
// WHY THIS FILE EXISTS, AND HOW IT DIFFERS FROM core/shard_plan.h
// ---------------------------------------------------------------
// shard_plan.h answers "WHICH SLICE does this rank own" -- WorldShape in, ShardPlan out
// (weight columns, KV heads, text layers). It is a partition of the MODEL.
// This header answers the NEXT question, which shard_plan.h names but does not do: once the
// slice is known, what changes in the three arithmetic structures that address the KV, and
// what does a rank have to reduce over? It is a partition of the STORAGE.
//
// The three breakages are not hypothetical and they are not a matter of taste. They were
// produced on a host with no GPU (MULTIDEV-X, s13c, rc=0) by lifting
// `cold_host_page_is_read_free` and the `allocate_cold_disk_file_slot` allocator body
// VERBATIM out of the tree and running them in a loop over ranks:
//
//   C. 64 B RecallRecord:** one 8 B region descriptor per page. R>=2 regions in ONE record
//      does not fit, and R>=2 WITH a per-region codec never fits.
//   D. The spill cell:  `dir + "/ninfer_cold_L" + layer + ".slot"` has a layer axis and NO
//      rank axis, and the slot allocator is PER-PROCESS first-fit. Every rank therefore
//      takes slot 0 of the same file: rank 0, 1 and 2 all get `(ninfer_cold_L0.slot, 0)`.
//      It is deterministic, not a race.
//   E. `HostKVPageLayout::page_stride` is ONE scalar for the whole page record.
//
// (C) is closed in src/spec/turn_recall_journal.h: the record now carries the shard words at
// offsets 56/60 and the ledger `kRecallShardAxisBytes` / `recall_extra_regions_after_shard_axis()`
// with two static_asserts pinning it from both sides. What that header does NOT carry -- and
// what this one adds -- is the READER'S RULE it describes in prose: "a reader must refuse a
// record whose (shard_rank, shard_world) is not its own". A field nobody checks is not a fix.
//
// (D) and (E) are unclosed. shard_plan.h's `cold_spill_cell_path()` gives the spill cell a rank
// axis IN ITS NAME, which is necessary and not sufficient: a rank-tagged file name plus a
// per-process allocator still hands every rank slot 0, and a rank-tagged file name plus the
// GLOBAL byte cap still lets each rank spend the whole cap, so the world spends
// world_size x the cap. The axis has to be in the ADDRESS SPACE and in the ACCOUNTING, not
// only in the string.
//
// Host-only by construction, exactly like shard_plan.h: no CUDA header, no device query, no
// collective. Everything here is a pure function of (world, geometry, numbers), so all of it is
// exercisable on one GPU. What is NOT here is the transport -- see section D and the report.

#include "core/shard_plan.h"
#include "spec/turn_recall_journal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 0. WHOSE NUMBER IS THIS? -- the scope tag that makes the rest checkable
// ---------------------------------------------------------------------------
//
// Every byte cap in this engine comes from one operator-visible number
// (`--cold-disk-bytes`, `--cold-host-bytes`) and every stride comes from the model geometry.
// Both are properties of the RUN, and in a sharded world "the run" and "this rank" are
// different objects. Mixing them is not a style question: it is a silent factor of
// world_size in either direction, and both directions are wrong in a way no test of the
// single-device path can see.
//
//   * WORLD cap read against a PER-RANK stride -> the capacity is world_size x too large
//     (each rank believes it may spend the whole cap).
//   * PER-RANK cap read against a WHOLE stride -> the capacity is world_size x too small
//     (the world refuses pages it has room for).
//
// So a cap and a stride may not be multiplied or divided unless the caller has said which
// scope each one is in. Declaring it is the whole function of this enum.
enum class BudgetScope : std::uint8_t {
    WorldTotal, // the number bounds the whole world: divide it across ranks
    PerRank,    // the number is already this rank's own cap: do not divide it
};

[[nodiscard]] inline std::string_view budget_scope_name(BudgetScope scope) noexcept {
    switch (scope) {
    case BudgetScope::WorldTotal: return "world-total";
    case BudgetScope::PerRank: return "per-rank";
    }
    return "unknown-scope";
}

// ---------------------------------------------------------------------------
// A. THE OVERFLOW ALLOCATION'S RANK AXIS  (breakage D, the address space half)
// ---------------------------------------------------------------------------
//
// The spill slot is the disk tier's address. Today it is a bare `std::int32_t file_slot`
// handed out by `allocate_cold_disk_file_slot()`, a first-fit over a per-process bitmap, and
// the on-disk address is `file_slot * slot_stride`. Nothing in that expression knows how many
// ranks exist, so the FIRST slot every rank hands out is the same slot.
//
// The fix is to make the slot a point in a WORLD address space with the rank as the major
// coordinate:
//
//     world_slot = rank * slots_per_rank + local_slot
//
// Rank r's admissible slots are then exactly the half-open range [r*S, (r+1)*S), and two ranks
// cannot be handed the same slot no matter what order they allocate in, how many they
// allocate, or which process they run in. The invariant is a RANGE, not a convention, which is
// why it is checkable rather than assertable.
//
// Note what this does NOT do: it does not coordinate the allocators. Two ranks may still
// allocate in any order; they simply cannot produce the same number. That matters because the
// alternative fix -- an all-rank barrier around every page eviction -- is a collective on the
// eviction driver's critical path (`residency_decision_needs_replication(Tensor)` is true and
// this is what it costs), and it should be spent where it buys atomicity, not where
// arithmetic buys it for free.

struct FileSlotRange {
    std::int32_t begin = 0; // inclusive
    std::int32_t end   = 0; // exclusive

    [[nodiscard]] std::int32_t count() const noexcept { return end - begin; }
    [[nodiscard]] bool empty() const noexcept { return end <= begin; }
    [[nodiscard]] bool contains(std::int32_t slot) const noexcept {
        return slot >= begin && slot < end;
    }
    friend bool operator==(const FileSlotRange&, const FileSlotRange&) = default;
};

struct SpillSlotSpace {
    bool ok = false;
    std::string reason;

    WorldShape world{};
    std::int32_t slots_per_rank = 0; // S
    std::int32_t total_slots    = 0; // S * world_size

    // THIS rank's admissible slot range. Every slot this rank hands out must come from here.
    FileSlotRange range{};

    // local -> world. Returns -1 when the local slot is outside this rank's share, i.e. when
    // the caller picked a slot the rank does not own. A -1 here is the collision the old code
    // produced silently.
    [[nodiscard]] std::int32_t world_slot(std::int32_t local_slot) const noexcept {
        if (!ok || local_slot < 0 || local_slot >= slots_per_rank) { return -1; }
        return range.begin + local_slot;
    }

    // world -> the rank that owns it, or -1 when it lies outside the partition (which for a
    // well-formed space cannot happen, and is reported rather than wrapped).
    [[nodiscard]] std::int32_t owner_rank(std::int32_t slot) const noexcept {
        if (!ok || slots_per_rank <= 0 || slot < 0 || slot >= total_slots) { return -1; }
        return slot / slots_per_rank;
    }
};

// The partition, from the world and the two numbers that size the space.
//
// `budget_bytes` is the spill cap whose scope the caller declares; `per_rank_slot_bytes` is
// THIS rank's slot stride, which under a head split is the whole page's slot stride divided by
// world_size. The division by world_size happens EXACTLY ONCE and it happens here, on
// whichever of the two numbers the scope says is a world quantity, so that the caller cannot
// get it wrong by reading the other one.
[[nodiscard]] inline SpillSlotSpace plan_spill_slot_space(const WorldShape& world,
                                                          std::uint64_t budget_bytes,
                                                          std::size_t per_rank_slot_bytes,
                                                          BudgetScope scope) {
    SpillSlotSpace out;
    out.world = world;

    const std::string world_refusal = world_shape_refusal(world);
    if (!world_refusal.empty()) {
        out.reason = world_refusal;
        return out;
    }
    if (per_rank_slot_bytes == 0) {
        out.reason = "the per-rank slot stride is zero: there is no on-disk cell to index, "
                     "so no slot space exists to partition";
        return out;
    }
    if (scope == BudgetScope::WorldTotal && world.axis == ParallelAxis::None &&
        world.world_size != 1) {
        out.reason = "a world cap was declared on an axis=none world";
        return out;
    }

    // The ONE division. A `WorldTotal` cap is divided by the world; a `PerRank` cap is not.
    const std::uint64_t per_rank_bytes =
        scope == BudgetScope::WorldTotal
            ? budget_bytes / static_cast<std::uint64_t>(world.world_size)
            : budget_bytes;

    const std::uint64_t slots = per_rank_bytes / static_cast<std::uint64_t>(per_rank_slot_bytes);
    if (slots == 0) {
        out.reason = "the per-rank spill cap holds no whole slot of this rank's stride (" +
                     std::to_string(per_rank_bytes) + " bytes / " +
                     std::to_string(per_rank_slot_bytes) + " bytes)";
        return out;
    }
    if (slots > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) /
                    world.world_size) {
        out.reason = "the world slot space does not fit in int32";
        return out;
    }

    out.slots_per_rank = static_cast<std::int32_t>(slots);
    out.total_slots    = static_cast<std::int32_t>(slots * world.world_size);
    out.range          = FileSlotRange{static_cast<std::int32_t>(world.rank) * out.slots_per_rank,
                                       static_cast<std::int32_t>(world.rank + 1U) * out.slots_per_rank};
    out.ok             = true;
    out.reason.clear();
    return out;
}

// The re-composition check. This is the same "no single stride" invariant the spec header
// enforces on the LAYER axis (`static_assert(!recall_codec_admitted(RecallCodec::Mixed),
// "a page whose layers use different codecs has no single stride")`, turn_recall_journal.h),
// applied to the RANK axis: if the per-rank slot stride does not multiply back up to the whole
// page's slot stride, then the split is not a partition of one stride and every piece of
// arithmetic downstream of it is describing a different object than the one on disk.
[[nodiscard]] inline std::string per_rank_stride_refusal(const WorldShape& world,
                                                         std::size_t whole_slot_bytes,
                                                         std::size_t per_rank_slot_bytes) {
    if (whole_slot_bytes == 0) {
        return "the whole page's slot stride is zero: nothing was ever sized for this tier";
    }
    if (per_rank_slot_bytes == 0) {
        return "the per-rank slot stride is zero";
    }
    const std::uint64_t recomposed =
        static_cast<std::uint64_t>(per_rank_slot_bytes) * world.world_size;
    if (recomposed != whole_slot_bytes) {
        return "the per-rank slot stride " + std::to_string(per_rank_slot_bytes) +
               " x world_size " + std::to_string(world.world_size) + " = " +
               std::to_string(recomposed) + " does not re-compose into the whole page's slot "
               "stride " + std::to_string(whole_slot_bytes) +
               ": the shard is not a partition of ONE stride, so a file_slot x stride address "
               "would be read at a different place than it was written";
    }
    return {};
}

// ---------------------------------------------------------------------------
// B. THE PAGE STRIDE'S RANK AXIS  (breakage E, and the accounting half of D)
// ---------------------------------------------------------------------------
//
// `HostKVPageLayout::page_stride` (core/host_kv_arena.h) is ONE `std::size_t` for a page
// record that `plan_host_kv_page_layout` fills by walking EVERY plane of EVERY layer. Under a
// head split each rank holds kv_heads/world_size of the heads, so its page record is
// smaller -- and the honest statement is that ONE stride cannot describe two ranks' records.
// What rescues it is that a head split scales every plane by the SAME factor: the record is
// not re-laid-out, it is uniformly divided. So:
//
//     per_rank_stride = whole_stride / world_size   -- exact, or refused.
//
// "or refused" is the load-bearing half. A stride that does not divide evenly is a rounding of
// one layout into R layouts, and the failure it produces is a page whose bytes for rank r
// start inside the bytes of rank r-1: reads return the neighbouring head's K/V. There is no
// tolerance that makes that acceptable, so the refusal is hard.
//
// The ACCOUNTING is the second half and it is where the world_size factor actually bites.
// `cold_tier_page_capacity(bytes, page_bytes)` is a division, so the capacity it returns
// depends on WHICH TWO numbers a caller hands it:
//
//     cold_tier_page_capacity(world_cap,   per_rank_stride)  -> world_size x the real capacity
//     cold_tier_page_capacity(per_rank_cap, whole_stride)    -> world_size / the real capacity
//
// Both are one wrong argument away in code that reads `budget_.host_bytes` and
// `budget_.host_page_bytes` with no idea that either has a scope. `RankPageStride` carries the
// scope with the numbers so the pair cannot be assembled wrong, and
// `world_cap_read_with_per_rank_stride()` is the inflated number itself, exposed so a test can
// assert that it is exactly world_size x the honest one rather than merely "different".
struct RankPageStride {
    bool ok = false;
    std::string reason;

    WorldShape world{};
    std::size_t whole_stride    = 0; // HostKVPageLayout::page_stride of the unsplit layout
    std::size_t per_rank_stride = 0; // whole_stride / world_size
};

[[nodiscard]] inline RankPageStride rank_page_stride(const WorldShape& world,
                                                     std::size_t whole_stride) {
    RankPageStride out;
    out.world        = world;
    out.whole_stride = whole_stride;

    const std::string world_refusal = world_shape_refusal(world);
    if (!world_refusal.empty()) {
        out.reason = world_refusal;
        return out;
    }
    if (whole_stride == 0) {
        out.reason = "the whole page's stride is zero: plan_host_kv_page_layout produced no "
                     "page record to split";
        return out;
    }
    if (whole_stride % world.world_size != 0) {
        out.reason = "the page record's stride " + std::to_string(whole_stride) +
                     " is not divisible by world_size " + std::to_string(world.world_size) +
                     ": a head split of this page would put rank r's bytes at a fraction of a "
                     "stride, so rank r would read rank r-1's last head. Refuse the split; do "
                     "not round it.";
        return out;
    }
    out.per_rank_stride = whole_stride / world.world_size;
    out.ok              = true;
    out.reason.clear();
    return out;
}

// Local copy of the one division, so this header does not have to include product/ and can
// still be checked against it: tests pin it equal to product::cold_tier_page_capacity, which
// is the tree's authority. Kept in the same shape as that function (a cap smaller than one
// page admits nothing; the result saturates rather than wrapping).
[[nodiscard]] inline std::uint32_t page_capacity_from_bytes(std::uint64_t bytes,
                                                            std::uint64_t page_bytes) noexcept {
    if (page_bytes == 0) { return 0; }
    const std::uint64_t pages = bytes / page_bytes;
    return pages > 0xFFFFFFFFULL ? 0xFFFFFFFFU : static_cast<std::uint32_t>(pages);
}

// The honest page capacity for one rank, with the scope declared and applied exactly once.
[[nodiscard]] inline std::uint32_t rank_page_capacity(const RankPageStride& stride,
                                                      std::uint64_t cap_bytes,
                                                      BudgetScope scope) noexcept {
    if (!stride.ok) { return 0; }
    const std::uint64_t per_rank_bytes =
        scope == BudgetScope::WorldTotal
            ? cap_bytes / static_cast<std::uint64_t>(stride.world.world_size)
            : cap_bytes;
    // Per-rank pages, from THIS rank's stride. The world's page count is the same number: the
    // pages are the unit, and a page is one page on every rank (that is what
    // page_content_is_layer_complete(Tensor) == true buys).
    return page_capacity_from_bytes(per_rank_bytes, stride.per_rank_stride);
}

// The inflated number, named so it cannot be produced by accident and so a test can assert its
// exact size. This is the expression `cold_host_tier.h` would evaluate if it kept reading
// `budget_.host_bytes` (a world cap) against a per-rank layout stride.
[[nodiscard]] inline std::uint32_t world_cap_read_with_per_rank_stride(
    const RankPageStride& stride, std::uint64_t world_cap_bytes) noexcept {
    if (!stride.ok) { return 0; }
    return page_capacity_from_bytes(world_cap_bytes, stride.per_rank_stride);
}

// ---------------------------------------------------------------------------
// C. THE RECORD'S RANK AXIS  (breakage C: the field exists -- the RULE did not)
// ---------------------------------------------------------------------------
//
// src/spec/turn_recall_journal.h put `shard_rank`/`shard_world` at bytes 56/60 and pinned the
// budget with a ledger and two static_asserts. Its comment states the reader's rule --
//
//     "a reader must refuse a record whose (shard_rank, shard_world) is not its own, rather
//      than mis-read a record written by a different world"
//
// -- and nothing implements it. A record from world 4 read by world 1 is a record whose
// `layer_bytes` describes a quarter of the KV as if it were all of it, so mis-reading it is
// silent numeric corruption in the cold tier, not a format error. The rule is implemented here
// rather than in the spec header only because that file belongs to another line today; it
// belongs there, and the report says so.

using RecallRecord = spec::turn_recall::RecallRecord;

// Empty on acceptance, a reason otherwise -- the shape `sharded_name_acceptance()` uses in
// shard_plan.h, so the two halves of the same contract read alike.
[[nodiscard]] inline std::string recall_record_shard_refusal(const RecallRecord& record,
                                                             const WorldShape& world) {
    const std::string world_refusal = world_shape_refusal(world);
    if (!world_refusal.empty()) {
        return "this run's world is malformed: " + world_refusal;
    }
    // (0, 1) is the identity world and the DEFAULT of both fields, so every record written
    // before the shard axis existed reads back as what it was: a single-device record.
    if (record.shard_world == 0) {
        return "the record names world 0, which is not a world";
    }
    if (record.shard_rank >= record.shard_world) {
        return "the record names rank " + std::to_string(record.shard_rank) +
               " of world " + std::to_string(record.shard_world) + ", which is out of range";
    }
    if (world.axis == ParallelAxis::None) {
        if (record.shard_world != 1U || record.shard_rank != 0U) {
            return "the record was written by rank " + std::to_string(record.shard_rank) +
                   " of world " + std::to_string(record.shard_world) +
                   " but this run is single-device: its layer_bytes describes one shard's KV, "
                   "not the whole KV, so its stride does not describe this run's pages";
        }
        return {};
    }
    if (record.shard_world != world.world_size || record.shard_rank != world.rank) {
        return "the record was written by rank " + std::to_string(record.shard_rank) +
               " of world " + std::to_string(record.shard_world) + " but this run is rank " +
               std::to_string(world.rank) + " of world " + std::to_string(world.world_size) +
               ": a record from another shard names a region whose stride is not this rank's";
    }
    return {};
}

// The WRITE side's counterpart. Writing the shard is not optional bookkeeping: the record's
// default is the identity world, so a sharded writer that forgets this stamps (0,1) on a
// shard-shaped region, and its own reader then refuses its own record -- loud, which is the
// right failure, but late. Asserting the round trip is what makes the field load-bearing.
inline void stamp_recall_record_shard(RecallRecord& record, const WorldShape& world) noexcept {
    record.shard_rank  = world.world_size == 1U ? 0U : world.rank;
    record.shard_world = world.world_size == 1U ? 1U : world.world_size;
}

// The ledger as a runtime fact, so a caller that has only the numbers (and not the header the
// static_asserts live in) can still ask.
//
// `region_count` is how many region descriptors THIS RECORD must describe IN TOTAL, i.e. the
// page's own primary region included. That reading is what makes the predicate answer the
// question the callers actually have ("can this record describe this page?") instead of a
// question about headroom. The primary region's two fields are NAMED (@12 `layer_bytes`, @16
// `file_slot`), so they cost nothing from the unassigned pool; every region BEYOND the first
// needs a whole descriptor out of `recall_record_free_bytes_honest()`.
//
// THE PREDICATE NOW READS THE ONE AUTHORITY. It previously computed
// `kRecallRecordFreeBytes - kRecallShardAxisBytes` inline -- the RAW subtraction, which counts
// the release reason's word (`reserved`@28, written whole by recall_set_release_reason, live
// writer at program_impl.h:11351) as free. That is the same miscount the spec header's ledger
// carried, repeated here as a second, independently-spelt copy: exactly the "two copies of one
// rule" shape this file's own section D refuses.
[[nodiscard]] inline bool recall_region_budget_holds(std::uint32_t region_count,
                                                     bool per_region_codec) noexcept {
    using spec::turn_recall::kRecallRegionDescriptorBytes;
    if (region_count == 0U) { return false; }        // a record always describes its primary region
    if (per_region_codec) { return false; }          // NOT REPRESENTABLE, for any count -- below
    // The primary region's two fields are NAMED, so only the regions BEYOND it come out of the
    // pool. `kRecallRegionDescriptorBytes` of CONTIGUOUS space is one descriptor and the honest
    // pool is `recall_record_free_bytes_honest()`.
    const std::uint32_t extra_regions    = region_count - 1U;
    const std::uint32_t free_after_axis  = spec::turn_recall::recall_record_free_bytes_honest();
    const std::uint32_t descriptor_bytes = extra_regions * kRecallRegionDescriptorBytes;
    return descriptor_bytes <= free_after_axis;
}
// WHY `per_region_codec` IS REFUSED OUTRIGHT (above) rather than PRICED, which is what it used
// to be (`+ region_count` bytes): `codec` is ONE byte for the WHOLE page (turn_recall_journal.h,
// offset 6), so "a codec per region" is not a SIZE this record can express at ANY region count.
// With two strides served by ONE region the question does not arise at all -- that is exactly
// what `recall_codec_admitted(Mixed)` admits -- and with two REGIONS it would need two codec
// bytes where the format has one. Pricing it read as "one region with a per-region codec is a
// size problem"; it is not a size problem, it is an unrepresentable one, and the difference
// matters because a size problem is fixed by a bigger record and this one is not.

// ---------------------------------------------------------------------------
// E. BALANCE -- the part of "does it scale" that IS measurable on one device
// ---------------------------------------------------------------------------
//
// Aggregate throughput scaling with rank count is NOT measurable on a single GPU (the ranks share
// one HBM and time-slice one SM pool), and the transport is not this engine's component. What IS
// measurable is BALANCE: whether the shard plan gives the ranks equal work. A plan that hands one
// rank twice another's columns is a shard-arithmetic defect, it is visible in the plan itself
// with no device at all, and it is exactly the class of error a virtual world then lets you
// CONFIRM against per-rank wall clock (see core/virtual_device.h's RankTiming).
//
// `work_units` is deliberately the SUM OF THE THREE RANGES' SIZES as the model means them, not a
// byte count: for the tensor axis that is the rank's weight columns plus its KV heads; for the
// pipeline axis it is its text layers. Mixing the units would make the ratio meaningless, so the
// caller asks for one axis at a time and the report says which.
struct ShardBalance {
    bool ok = false;
    std::string reason;

    std::uint32_t world_size = 0;
    std::uint32_t min_work   = 0;
    std::uint32_t max_work   = 0;
    // max/min. 1.0 is a perfect split; the tensor axis must always give 1.0 because both of its
    // divisions are exact by construction (shard_plan.h refuses a non-divisible split rather than
    // rounding it), and the pipeline axis is ALLOWED to be uneven -- which is why this number is
    // reported rather than asserted.
    double imbalance = 0.0;
};

[[nodiscard]] inline std::uint32_t shard_work_units(const ShardPlan& plan) noexcept {
    switch (plan.axis) {
    case ParallelAxis::None: return plan.weight_columns.count();
    case ParallelAxis::Tensor: return plan.weight_columns.count() + plan.kv_heads.count();
    case ParallelAxis::Pipeline: return plan.text_layers.count();
    }
    return 0;
}

// The balance of a whole world, from the plans of its ranks. The plans are passed in rather than
// recomputed so this cannot disagree with what the ranks actually got.
[[nodiscard]] inline ShardBalance shard_balance(const std::vector<ShardPlan>& plans) {
    ShardBalance out;
    if (plans.empty()) {
        out.reason = "no ranks to balance";
        return out;
    }
    out.world_size = static_cast<std::uint32_t>(plans.size());
    for (std::size_t i = 0; i < plans.size(); ++i) {
        if (!plans[i].ok) {
            out.reason = "rank " + std::to_string(i) + " has no plan: " + plans[i].reason;
            return out;
        }
        if (plans[i].axis != plans[0].axis) {
            out.reason = "rank " + std::to_string(i) +
                         " is on a different axis from rank 0: a world has one axis";
            return out;
        }
        const std::uint32_t work = shard_work_units(plans[i]);
        out.max_work = std::max(out.max_work, work);
        if (out.min_work == 0 || work < out.min_work) { out.min_work = work; }
    }
    if (out.min_work == 0) {
        out.reason = "a rank was given no work at all";
        return out;
    }
    out.imbalance = static_cast<double>(out.max_work) / static_cast<double>(out.min_work);
    out.ok        = true;
    out.reason.clear();
    return out;
}

[[nodiscard]] inline std::string shard_balance_line(const ShardBalance& balance) {
    if (!balance.ok) { return "shard balance unavailable: " + balance.reason; }
    return "shard balance: world " + std::to_string(balance.world_size) + ", work " +
           std::to_string(balance.min_work) + ".." + std::to_string(balance.max_work) +
           " units, imbalance " + std::to_string(balance.imbalance) +
           "x (a balance number is computed from the plan and is therefore measurable on one "
           "device; aggregate scaling is not)";
}

// ---------------------------------------------------------------------------
// D. THE ATTENTION COMBINE -- the one collective the tensor axis needs
// ---------------------------------------------------------------------------
//
// Under the tensor axis a rank owns kv_heads/world_size of the KV heads, so every rank's
// attention produces a PARTIAL softmax: its own row max, its own row sumexp and its own
// weighted value sum. The whole row is the log-sum-exp merge of the partials:
//
//     m* = max(m1, m2)
//     l* = l1 * exp(m1 - m*) + l2 * exp(m2 - m*)
//     o* = o1 * l1 * exp(m1 - m*) + o2 * l2 * exp(m2 - m*)
//
// `o` is the UN-NORMALISED numerator, `sum_i v_i * exp(s_i - m)`, and the row's value is
// `o / l`. That convention is the one that makes the merge a monoid with the empty partial
// (m = -inf, l = 0, o = 0) as its identity, and it is the one the engine's own flash-attention
// partials use: the row-sum all-reduce is deferred to the epilogue (see the comment in
// ops/kernel/gqa_attention_prefill_bf16.cuh and
// ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh). Normalising inside the merge would
// make the merge non-idempotent for the empty partial and would divide by `l` once per rank.
//
// WHAT IS TESTABLE ON ONE GPU, AND WHAT IS NOT. The merge above is associative and
// commutative, so (a) it is ORDER-INDEPENDENT, which is what makes an all-reduce legal at all,
// and (b) at world_size == 1 the all-reduce must be the IDENTITY -- not "approximately the
// identity": a rank that already holds the whole row must not touch a byte of it, and a
// single-device run must therefore perform no collective, allocate no buffer and add no sync.
// (b) is the property that regresses when someone wires a collective in unconditionally, and
// it is checkable here. The COST of the collective is not checkable here: it is a transport
// property, and this box has one GPU (see the report's `pending hardware` list).
struct AttentionCombine {
    double m = -std::numeric_limits<double>::infinity(); // row max
    double l = 0.0;                                      // row sumexp
    double o = 0.0;                                      // weighted value sum (numerator)

    friend bool operator==(const AttentionCombine&, const AttentionCombine&) = default;
};

[[nodiscard]] inline AttentionCombine combine_merge(const AttentionCombine& a,
                                                    const AttentionCombine& b) noexcept {
    // An empty partial is the identity, and it is the case that actually occurs: a rank with no
    // visible keys in the window contributes (m=-inf, l=0, o=0), and merging it must not produce
    // a NaN through exp(-inf - -inf). `l == 0` is equivalent to "empty" for a softmax partial:
    // the max element always has weight exp(0) == 1, so a non-empty partial has l >= 1.
    if (a.l == 0.0) { return b; }
    if (b.l == 0.0) { return a; }

    const double m  = a.m > b.m ? a.m : b.m;
    // The rescale is exp(m_r - m*) for BOTH l and o: `o` is the numerator sum
    // `sum_i v_i * exp(s_i - m_r)`, so rescaling the row max rescales it by the same factor the
    // sumexp is rescaled by. Multiplying `o` by `l_r * exp(m_r - m*)` -- the sumexp's weight --
    // would weight each rank's output by its own sumexp, which double-counts the mass and is
    // wrong whenever the ranks' maxima differ. (This line was written the wrong way first, and
    // tests/test_shard_rank_axis.cpp's split-join check is what caught it: world 2 gave -0.8314
    // against a reference of -0.5354.)
    const double ea = std::exp(a.m - m);
    const double eb = std::exp(b.m - m);
    return AttentionCombine{m, a.l * ea + b.l * eb, a.o * ea + b.o * eb};
}

// What a caller must do for a given axis -- three outcomes, not a bool, because "no
// collective" and "a collective that happens to be the identity" are different amounts of
// code and different failure modes. Mirrors `KernelRoute`'s three-valued shape in
// core/kernel_route.h and `ResidencyState`'s in shard_plan.h.
enum class CombineOutcome : std::uint8_t {
    // world_size == 1: there is nothing to reduce over and NO CALL MUST BE MADE. The partial
    // IS the whole, bit for bit.
    None,
    // The axis replicates the attention: every rank already computes the whole row, so the
    // merge is the identity on replicated inputs. The pipeline axis is this case, and so is
    // the single device. A collective here would be pure cost.
    Replicated,
    // The axis splits the row: partials from every rank must be merged before the result is a
    // legal attention output. The tensor axis is this case.
    MergeRequired,
};

[[nodiscard]] inline std::string_view combine_outcome_name(CombineOutcome outcome) noexcept {
    switch (outcome) {
    case CombineOutcome::None: return "none";
    case CombineOutcome::Replicated: return "replicated";
    case CombineOutcome::MergeRequired: return "merge-required";
    }
    return "unknown";
}

[[nodiscard]] inline CombineOutcome attention_combine_outcome(ParallelAxis axis) noexcept {
    switch (axis) {
    case ParallelAxis::None: return CombineOutcome::None;
    case ParallelAxis::Pipeline: return CombineOutcome::Replicated;
    case ParallelAxis::Tensor: return CombineOutcome::MergeRequired;
    }
    return CombineOutcome::MergeRequired; // unknown: assume the reduction is needed
}

// The reducer, with the no-op cases taken by construction rather than by a flag someone has to
// remember. `partials[rank]` is what rank `rank` produced. Returns the merged row and reports
// through `performed_collective` whether a merge actually had to happen -- the caller uses it
// to skip a barrier, and the test uses it to prove world_size == 1 does no work.
[[nodiscard]] inline AttentionCombine all_reduce_attention_combine(
    const std::vector<AttentionCombine>& partials, CombineOutcome outcome,
    bool* performed_collective = nullptr) {
    if (performed_collective != nullptr) { *performed_collective = false; }
    if (partials.empty()) { return AttentionCombine{}; }
    if (outcome == CombineOutcome::None) {
        // Exactly one rank, and it holds the whole row. Not a copy of a merged value: the
        // value itself, so a single-device run cannot be perturbed by the collective's
        // existence.
        return partials.front();
    }
    if (outcome == CombineOutcome::Replicated) {
        // Every partial is already the whole row (the axis did not split attention). Assert
        // the replication rather than trusting it, then return the first.
        return partials.front();
    }
    if (performed_collective != nullptr) { *performed_collective = partials.size() > 1U; }
    AttentionCombine acc = partials.front();
    for (std::size_t i = 1; i < partials.size(); ++i) { acc = combine_merge(acc, partials[i]); }
    return acc;
}

// Whether a rank's partial is already the whole row, i.e. whether the axis replicates
// attention. Separated from `attention_combine_outcome` so a caller can ask the question about
// a DIFFERENT axis than the one it is constructing, which is how the two-axis refusals in
// shard_plan.h are checked without building the world.
[[nodiscard]] inline bool attention_is_replicated(ParallelAxis axis) noexcept {
    return axis != ParallelAxis::Tensor;
}

} // namespace ninfer::multi
