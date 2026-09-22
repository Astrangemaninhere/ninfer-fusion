// test_shard_rank_axis.cpp -- the rank axis, exercised on a host with no GPU.
//
// This file exists because a plan is not progress. MULTIDEV-X produced the counterexample that
// tells a plan apart from progress: it lifted `cold_host_page_is_read_free` and the
// `allocate_cold_disk_file_slot` allocator body VERBATIM out of the tree, ran them in a loop over
// ranks, and got rc=0 and this output --
//
//     R=1 -> fits
//     R>=2 -> DOES NOT FIT in the 64 B record (kRecallRecordBytes)
//     rank 0 own-bitmap : file=.../ninfer_cold_L0.slot file_slot=0 offset=0
//     rank 1 own-bitmap : file=.../ninfer_cold_L0.slot file_slot=0 offset=0
//     rank 2 own-bitmap : file=.../ninfer_cold_L0.slot file_slot=0 offset=0
//
// -- i.e. R ranks deterministically resolving to the SAME (file, offset), because the allocator's
// free list is per-process and its address space has no rank dimension.
//
// THE POINT OF THIS FILE IS THAT IT SHOWS THE OPPOSITE, and shows it the hard way: the broken
// allocator is re-implemented here and REQUIRED to collide (group A1), so that the same loop over
// the partitioned space (group A2) proves something rather than passing by luck.
//
// Host-only: no CUDA header, no device, no GPU, no artifact.

#include "core/shard_rank_axis.h"
#include "core/shard_plan.h"
#include "core/virtual_device.h"
#include "product/kv_bit_budget.h"       // kKvBitBudgetColdSlotBytes: the cold stride, DERIVED
#include "product/kv_cold_tier_budget.h" // the tree's authority for the one division
#include "spec/turn_recall_journal.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::multi::all_reduce_attention_combine;
using ninfer::multi::AttentionCombine;
using ninfer::multi::attention_combine_outcome;
using ninfer::multi::attention_is_replicated;
using ninfer::multi::BudgetScope;
using ninfer::multi::CombineOutcome;
using ninfer::multi::combine_merge;
using ninfer::multi::FileSlotRange;
using ninfer::multi::page_capacity_from_bytes;
using ninfer::multi::ParallelAxis;
using ninfer::multi::per_rank_stride_refusal;
using ninfer::multi::plan_spill_slot_space;
using ninfer::multi::rank_page_capacity;
using ninfer::multi::rank_page_stride;
using ninfer::multi::RankPageStride;

// The tree's cold slot stride for ONE layer, TAKEN FROM THE AUTHORITY rather than
// re-typed here: product/kv_bit_budget.h's kKvBitBudgetColdSlotBytes, which is itself
// derived from product/kv_tier_formats.h's cold record (kKvColdPoolStrideBytes) and,
// through it, from decoder_state.cpp's kColdSlotRansBitsPerCodeX100 -- a MEASURED
// ceiling. The literal this file used to carry was 6688 B, and it did not move when
// the record did: the rANS record has been 9632 B since 2026-09-18, so a re-typed
// stride sat 2944 B stale and NOTHING IN THIS FILE FAILED. Reading it from the
// header is what stops the fourth half-rename: a stride this file cannot re-type is
// a stride this file cannot get wrong.
constexpr std::uint32_t kNvfp4ColdStride =
    static_cast<std::uint32_t>(ninfer::product::kKvBitBudgetColdSlotBytes);
using ninfer::multi::recall_record_shard_refusal;
using ninfer::multi::recall_region_budget_holds;
using ninfer::multi::ShardPlan;
using ninfer::multi::shard_balance;
using ninfer::multi::shard_balance_line;
using ninfer::multi::shard_work_units;
using ninfer::multi::SpillSlotSpace;
using ninfer::multi::stamp_recall_record_shard;
using ninfer::multi::world_cap_read_with_per_rank_stride;
using ninfer::multi::WorldShape;

int failures = 0;
int checks   = 0;
int reported = 0;
constexpr int kReportLimit = 30;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        if (reported < kReportLimit) {
            ++reported;
            std::cout << "FAIL: " << what << "\n";
        } else if (reported == kReportLimit) {
            ++reported;
            std::cout << "FAIL: ... further failures suppressed\n";
        }
    }
}

[[nodiscard]] WorldShape tp(std::uint32_t world, std::uint32_t rank) {
    return WorldShape{world, rank, ParallelAxis::Tensor};
}

// ---------------------------------------------------------------------------
// A. THE OVERFLOW ALLOCATION'S RANK AXIS
// ---------------------------------------------------------------------------
//
// THE BROKEN ALLOCATOR, transcribed from program_impl.h's `allocate_cold_disk_file_slot()`
// (first-fit over a per-process `cold_disk_file_used` bitmap, returning the lowest free index) and
// from the call site's path expression (`dir + "/ninfer_cold_L" + layer + ".slot"`). It is here to
// be run and to FAIL: a check whose subject is "this is fixed" is worth nothing until the
// unfixed form has been shown to produce the defect.
struct PerProcessAllocator {
    std::vector<std::uint8_t> used;
    explicit PerProcessAllocator(std::size_t slots) : used(slots, 0) {}
    std::int32_t allocate() noexcept {
        for (std::size_t slot = 0; slot < used.size(); ++slot) {
            if (used[slot] == 0) {
                used[slot] = 1;
                return static_cast<std::int32_t>(slot);
            }
        }
        return -1;
    }
};

[[nodiscard]] std::string rank_blind_cell(std::string_view dir, std::uint32_t layer) {
    // Byte-for-byte the tree's current expression. `rank` is deliberately not a parameter.
    return std::string(dir) + "/ninfer_cold_L" + std::to_string(layer) + ".slot";
}

using Cell = std::pair<std::string, std::int32_t>; // (file, file_slot) == (file, offset)

void test_a1_the_unpartitioned_space_must_collide() {
    // Three ranks, each with its OWN process (so: its own allocator), each taking its first slot.
    std::vector<Cell> cells;
    for (std::uint32_t rank = 0; rank < 3; ++rank) {
        PerProcessAllocator allocator(8); // per-process, exactly as the tree's is
        (void)rank;
        const std::int32_t file_slot = allocator.allocate();
        cells.emplace_back(rank_blind_cell("/tmp", 0), file_slot);
    }
    std::set<Cell> distinct(cells.begin(), cells.end());
    check(distinct.size() == 1U,
          "the tree's address space must collapse every rank onto ONE (file, file_slot): got " +
              std::to_string(distinct.size()) + " distinct cells for 3 ranks");
    check(cells[0].second == 0 && cells[1].second == 0 && cells[2].second == 0,
          "every rank's first slot must be 0 -- this is arithmetic, not a race");
    check(cells[0].first == "/tmp/ninfer_cold_L0.slot",
          "and the file name must be the rank-blind one the tree spells today");
    // The control for the control: the collision is not caused by the file name alone. Even with
    // PER-RANK FILES, a shared slot space gives each rank offset 0 in its own file -- which is
    // safe on disk but still means the SLOT NUMBER is not an address. The rank axis has to be in
    // both, and that is what group A2 asserts.
    std::vector<Cell> per_rank_file_shared_slots;
    for (std::uint32_t rank = 0; rank < 3; ++rank) {
        PerProcessAllocator allocator(8);
        per_rank_file_shared_slots.emplace_back(
            "/tmp/ninfer_cold_L0.r" + std::to_string(rank) + "of3.slot", allocator.allocate());
    }
    std::set<Cell> distinct_files(per_rank_file_shared_slots.begin(),
                                  per_rank_file_shared_slots.end());
    check(distinct_files.size() == 3U,
          "per-rank file names DO separate the cells -- which is why shard_plan.h's "
          "cold_spill_cell_path() is necessary. A2 shows it is not sufficient for the accounting.");
}

void test_a2_the_partitioned_space_must_not_collide() {
    // The same allocator, the same loop, the same "take the first free slot" -- but each rank
    // allocates inside the range the partition hands it. This is the whole fix.
    for (const std::uint32_t world : {2U, 3U, 4U, 8U}) {
        std::set<Cell> cells;
        std::set<std::int32_t> world_slots;
        std::int32_t expected_total = 0;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const auto space = plan_spill_slot_space(
                tp(world, rank), /*budget_bytes=*/64ULL * 1024 * 1024,
                /*per_rank_slot_bytes=*/4096, BudgetScope::WorldTotal);
            check(space.ok, "the partition must exist for world " + std::to_string(world) +
                                " rank " + std::to_string(rank) + ": " + space.reason);
            if (!space.ok) { continue; }
            expected_total = space.total_slots;

            // Each rank owns its OWN allocator (as it does: one process per rank), but the slots
            // it may hand out are local indices inside its range.
            PerProcessAllocator local(static_cast<std::size_t>(space.slots_per_rank));
            const std::int32_t local_slot = local.allocate();
            const std::int32_t slot       = space.world_slot(local_slot);
            check(slot >= 0, "a local slot inside the rank's share must map to a world slot");
            check(space.range.contains(slot),
                  "rank " + std::to_string(rank) + "'s slot must lie in its own range");
            check(space.owner_rank(slot) == static_cast<std::int32_t>(rank),
                  "the world slot must name the rank that owns it");

            // The cell: the rank's own file AND the rank's own slot. Both axes, which is the
            // combination A1 showed is required.
            const Cell cell{"/tmp/ninfer_cold_L0.r" + std::to_string(rank) + "of" +
                                std::to_string(world) + ".slot",
                            slot};
            check(cells.insert(cell).second,
                  "two ranks produced the same (file, file_slot): " + cell.first + " @ " +
                      std::to_string(slot));
            check(world_slots.insert(slot).second,
                  "two ranks produced the same world file_slot " + std::to_string(slot));
        }
        check(cells.size() == static_cast<std::size_t>(world),
              "world " + std::to_string(world) + " must give " + std::to_string(world) +
                  " distinct cells, got " + std::to_string(cells.size()));
        // Each rank's FIRST allocation must be the first slot of its own range: the partition is
        // what stops two ranks agreeing, and the first slot is where the old code collided.
        std::set<std::int32_t> expected_firsts;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const auto space = plan_spill_slot_space(tp(world, rank), 64ULL * 1024 * 1024, 4096,
                                                     BudgetScope::WorldTotal);
            if (space.ok) { expected_firsts.insert(space.range.begin); }
        }
        check(world_slots == expected_firsts,
              "the slots the ranks actually take must be exactly the first slot of each rank's "
              "range -- that is what makes the collision impossible by construction rather than "
              "by luck of allocation order");

        // And the ranges tile the world: no gap, no overlap, and the union is the whole space.
        std::int32_t covered = 0;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const auto space = plan_spill_slot_space(tp(world, rank), 64ULL * 1024 * 1024, 4096,
                                                     BudgetScope::WorldTotal);
            if (!space.ok) { continue; }
            check(space.range.begin == covered,
                  "rank " + std::to_string(rank) + "'s share must start where rank " +
                      std::to_string(rank - 1) + "'s ended");
            covered = space.range.end;
        }
        check(covered == expected_total, "the ranges must cover the whole slot space exactly");
    }
}

void test_a3_the_accounting_is_divided_exactly_once() {
    // The cap is a WORLD quantity (it bounds the run) and the stride is a PER-RANK quantity
    // (each rank holds 1/world of the page). If both were divided -- or neither -- the world would
    // reserve the wrong amount of disk, and the error is a factor of world_size in one direction
    // or the other.
    constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    const std::uint64_t world_cap = 8ULL * kGiB;

    // A 2 GiB per-rank slot stride, i.e. an 4 GiB whole-page stride for world 2.
    const auto world2_rank0 = plan_spill_slot_space(tp(2, 0), world_cap, 2ULL * kGiB,
                                                    BudgetScope::WorldTotal);
    check(world2_rank0.ok, "the world 2 partition must exist: " + world2_rank0.reason);
    check(world2_rank0.slots_per_rank == 2,
          "a world cap of 8 GiB over 2 ranks is 4 GiB per rank, which at a 2 GiB per-rank stride "
          "is 2 slots; got " + std::to_string(world2_rank0.slots_per_rank));
    check(world2_rank0.total_slots == 4,
          "the world space is therefore 4 slots, got " + std::to_string(world2_rank0.total_slots));

    // The WORLD's spend is capped: 4 slots x 2 GiB per rank = 8 GiB = the cap. Neither rank may
    // exceed it, and the two ranks together may not either.
    const std::uint64_t world_spend =
        static_cast<std::uint64_t>(world2_rank0.total_slots) * (2ULL * kGiB);
    check(world_spend <= world_cap,
          "the world's slot space must fit inside the world cap: " + std::to_string(world_spend) +
              " <= " + std::to_string(world_cap));

    // The OTHER scope: the same number declared per-rank means each rank gets the whole 8 GiB.
    const auto per_rank = plan_spill_slot_space(tp(2, 1), world_cap, 2ULL * kGiB,
                                                BudgetScope::PerRank);
    check(per_rank.ok, "the per-rank scope must be a legal declaration: " + per_rank.reason);
    check(per_rank.slots_per_rank == 4,
          "declaring the cap per-rank gives each rank 4 slots -- 16 GiB across the world, twice "
          "the 8 GiB the operator asked for. THIS is what the scope tag makes visible; got " +
              std::to_string(per_rank.slots_per_rank));
    check(static_cast<std::uint64_t>(per_rank.total_slots) * (2ULL * kGiB) == 2ULL * world_cap,
          "and the difference is exactly a factor of world_size");

    // Refusals. Each is a distinct arithmetic fact, not a generic failure.
    const auto zero_world = plan_spill_slot_space(WorldShape{0, 0, ParallelAxis::None}, world_cap,
                                                 4096, BudgetScope::WorldTotal);
    check(!zero_world.ok && !zero_world.reason.empty(), "world_size 0 must be refused");
    const auto bad_rank = plan_spill_slot_space(tp(2, 2), world_cap, 4096, BudgetScope::WorldTotal);
    check(!bad_rank.ok && !bad_rank.reason.empty(), "rank >= world_size must be refused");
    const auto zero_stride = plan_spill_slot_space(tp(2, 0), world_cap, 0,
                                                  BudgetScope::WorldTotal);
    check(!zero_stride.ok, "a zero per-rank slot stride must be refused");
    const auto too_small = plan_spill_slot_space(tp(8, 0), 4096, 4096, BudgetScope::WorldTotal);
    check(!too_small.ok,
          "a world cap that leaves a rank less than one slot must be refused rather than rounded "
          "to one");
    const auto unstated_axis =
        plan_spill_slot_space(WorldShape{4, 0, ParallelAxis::None}, world_cap, 4096,
                              BudgetScope::WorldTotal);
    check(!unstated_axis.ok, "world_size 4 with axis=none must be refused");

    // The stride re-composition: the "no single stride" invariant on the RANK axis.
    check(per_rank_stride_refusal(tp(4, 0), /*whole=*/kNvfp4ColdStride * 4,
                                  /*per_rank=*/kNvfp4ColdStride)
              .empty(),
          "a per-rank stride that multiplies back up to the whole must be accepted");
    const std::string ragged =
        per_rank_stride_refusal(tp(4, 0), kNvfp4ColdStride * 4, kNvfp4ColdStride - 1);
    check(!ragged.empty(),
          "a per-rank stride that does NOT re-compose must be refused: this is the same "
          "'no single stride' wall spec/turn_recall_journal.h raises on the layer axis");
    check(ragged.find("does not re-compose") != std::string::npos,
          "and the refusal must name the arithmetic, got: " + ragged);
}

// ---------------------------------------------------------------------------
// B. THE PAGE STRIDE'S RANK AXIS
// ---------------------------------------------------------------------------

void test_b1_the_page_stride_must_divide_exactly() {
    // kNvfp4ColdStride is the tree's cold slot stride for one layer -- 9632 B at the record
    // pinned on 2026-09-18, NOT the 6688 B this comment used to assert -- and it is read
    // from product/kv_bit_budget.h above rather than written here. 8 is qwen3_6_27b's
    // kv_heads, so a world of 2, 4 or 8 divides it exactly. 3 does not divide 8, and that is
    // the case that must be refused rather than rounded.
    for (const std::uint32_t world : {2U, 4U, 8U}) {
        const auto stride = rank_page_stride(tp(world, 0), kNvfp4ColdStride * 8U);
        check(stride.ok, "world " + std::to_string(world) + " must divide 8 heads: " +
                             stride.reason);
        check(stride.per_rank_stride == kNvfp4ColdStride * (8U / world),
              "the per-rank stride must be kv_heads/world times the per-head stride");
        check(stride.per_rank_stride * world == stride.whole_stride,
              "and it must re-compose exactly");
    }
    const auto ragged = rank_page_stride(tp(3, 0), kNvfp4ColdStride * 8U);
    check(!ragged.ok,
          "world 3 does not divide 8 KV heads, so a page record's stride has no exact third: "
          "the split must be refused, not rounded");
    check(ragged.reason.find("divisible") != std::string::npos,
          "and the refusal must say so, got: " + ragged.reason);
    const auto single =
        rank_page_stride(WorldShape{1, 0, ParallelAxis::None}, kNvfp4ColdStride);
    check(single.ok && single.per_rank_stride == kNvfp4ColdStride,
          "the identity world must leave the stride untouched");
    const auto zero = rank_page_stride(WorldShape{1, 0, ParallelAxis::None}, 0);
    check(!zero.ok, "a zero stride must be refused");
}

void test_b2_the_capacity_is_off_by_exactly_world_size_when_scopes_mix() {
    // The named defect. `cold_tier_page_capacity(bytes, page_bytes)` is a division, so the
    // capacity depends on WHICH two numbers a caller hands it, and nothing in the types says
    // which scope either one is in. That is the whole reason BudgetScope exists.
    constexpr std::uint64_t kMiB             = 1024ULL * 1024ULL;
    const std::uint64_t     world_host_bytes = 512ULL * kMiB;
    // STAND-IN value, deliberately not re-pointed at kKvBitBudgetColdSlotBytes (9632 B):
    // the equality checked at `undercounted == honest / world` is value-dependent, so the
    // stride here is an input this test chose, not a number mirrored from the tree.
    const std::size_t       whole_stride     = 6688U * 8U; // the whole page record

    for (const std::uint32_t world : {1U, 2U, 4U, 8U}) {
        const auto stride = rank_page_stride(
            world == 1U ? WorldShape{1, 0, ParallelAxis::None} : tp(world, 0), whole_stride);
        check(stride.ok, "the stride must split for world " + std::to_string(world));

        // HONEST: a world cap divided across ranks, then measured against the per-rank stride.
        const std::uint32_t honest =
            rank_page_capacity(stride, world_host_bytes, BudgetScope::WorldTotal);

        // WRONG: the world cap read against the per-rank stride -- one wrong argument away, and
        // the expression a caller writes by default when it has a budget and a layout in hand.
        const std::uint32_t inflated = world_cap_read_with_per_rank_stride(stride, world_host_bytes);

        // The relation is exact up to the truncation of one page per rank: the capacity is a
        // floor division, so "exactly world_size x" holds up to world_size - 1 pages. Stated as
        // the bound rather than as an equality, because asserting the equality would be asserting
        // something about the division that is false.
        check(inflated >= honest * world && inflated - honest * world < world,
              "the mixed-scope read must be world_size x the honest capacity to within the "
              "floor division's own truncation, for world " + std::to_string(world) +
                  " (honest " + std::to_string(honest) + ", mixed " + std::to_string(inflated) +
                  ")");
        check(world == 1U || inflated > honest,
              "for any split world the mixed read must be the LARGER number: over-counting "
              "capacity is the direction that admits pages the tier cannot hold");
        // And the third combination, the one that UNDER-counts.
        const std::uint32_t undercounted =
            page_capacity_from_bytes(world_host_bytes / world, whole_stride);
        check(undercounted == honest / world,
              "a per-rank cap read against the WHOLE stride must be world_size x too small (as a "
              "floor division): got " + std::to_string(undercounted) + ", want " +
                  std::to_string(honest / world));
        check(undercounted * world <= honest,
              "and it must never exceed the honest capacity -- under-counting refuses pages the "
              "world has room for, which is the mirror-image defect");
    }

    // The cross-check that makes those numbers authoritative rather than self-consistent: the one
    // division in this header must agree with the tree's own `cold_tier_page_capacity`, which is
    // what the engine actually calls. If this ever diverges, the fix above is describing a
    // different function than the engine runs.
    const std::vector<std::uint64_t> byte_samples{std::uint64_t{0},   std::uint64_t{1},
                                                 std::uint64_t{4096}, 100ULL * kMiB,
                                                 world_host_bytes};
    const std::vector<std::uint64_t> page_samples{std::uint64_t{1}, std::uint64_t{6688},
                                                  std::uint64_t{4096},
                                                  static_cast<std::uint64_t>(whole_stride)};
    for (const std::uint64_t bytes : byte_samples) {
        for (const std::uint64_t page : page_samples) {
            check(page_capacity_from_bytes(bytes, page) ==
                      ninfer::product::cold_tier_page_capacity(bytes, page),
                  "the local division must equal product::cold_tier_page_capacity at (" +
                      std::to_string(bytes) + ", " + std::to_string(page) + ")");
        }
    }
}

// ---------------------------------------------------------------------------
// C. THE RECORD'S RANK AXIS
// ---------------------------------------------------------------------------
//
// The field exists (spec/turn_recall_journal.h, bytes 56/60) and the budget is pinned from both
// sides by a ledger with two static_asserts. What did NOT exist is the rule the header describes
// in prose: "a reader must refuse a record whose (shard_rank, shard_world) is not its own". This
// group is that rule, and its red control.
void test_c1_the_region_budget_ledger() {
    check(recall_region_budget_holds(1, false),
          "one region with no per-region codec must fit after the shard axis");
    check(!recall_region_budget_holds(2, false),
          "two regions must NOT fit: R>=2 in one record is what MULTIDEV-X's rc=0 run proved "
          "arithmetically");
    check(!recall_region_budget_holds(1, true),
          "one region WITH a per-region codec must NOT fit: a page with two strides is the wall "
          "the spec header's own static_assert raises");
    // Red control: the ledger must be able to fail. If a second region with its own codec could
    // fit, the checks above would be vacuous.
    const std::uint32_t needs = 2U * ninfer::spec::turn_recall::kRecallRegionDescriptorBytes + 1U;
    // The ONE AUTHORITY, not the raw subtraction: the previous form read
    // `kRecallRecordFreeBytes - kRecallShardAxisBytes`, which counts `reserved`@28 -- the
    // release reason's word -- as free.
    const std::uint32_t has   = ninfer::spec::turn_recall::recall_record_free_bytes_honest();
    check(needs > has, "R=2 with a per-region codec must not fit (needs " + std::to_string(needs) +
                           ", has " + std::to_string(has) + ")");
    check(ninfer::spec::turn_recall::recall_extra_regions_after_shard_axis() == 0U,
          "and the ledger must leave NO extra region: the honest pool is 4 bytes and a "
          "descriptor is 8 CONTIGUOUS ones, so the record holds the primary region alone");
}

void test_c2_the_reader_must_refuse_another_worlds_record() {
    using ninfer::spec::turn_recall::RecallRecord;
    const WorldShape single{1, 0, ParallelAxis::None};
    const WorldShape w2r1 = tp(2, 1);

    // The default record is the identity world, so every record on disk today still reads back as
    // what it was: a single-device record. That is why the fields default to (0, 1) rather than to
    // an invalid sentinel.
    const RecallRecord fresh{};
    check(fresh.shard_rank == 0U && fresh.shard_world == 1U,
          "the default shard must be (0, 1)");
    check(recall_record_shard_refusal(fresh, single).empty(),
          "a (0,1) record read by a single-device run must be ACCEPTED -- otherwise this fix "
          "would refuse every record the engine has already written");

    // The defect: a sharded record read by a single-device run. Its layer_bytes describes one
    // shard's KV, so the stride does not describe this run's pages, and the mis-read is silent.
    RecallRecord sharded = fresh;
    stamp_recall_record_shard(sharded, w2r1);
    check(sharded.shard_rank == 1U && sharded.shard_world == 2U,
          "stamping a world 2 rank 1 record must write (1, 2)");
    const std::string wrong_world = recall_record_shard_refusal(sharded, single);
    check(!wrong_world.empty(),
          "a sharded record read by a single-device run MUST be refused, and the old code could "
          "not refuse it because there was no field to check");
    check(wrong_world.find("layer_bytes") != std::string::npos,
          "and the refusal must say WHY it matters (the stride is one shard's, not the whole "
          "KV's), got: " + wrong_world);

    // The round trip: the matching reader accepts, and every other reader refuses.
    check(recall_record_shard_refusal(sharded, w2r1).empty(),
          "the world that wrote it must accept its own record");
    for (const WorldShape other : {tp(2, 0), tp(4, 1), tp(4, 0)}) {
        const std::string reason = recall_record_shard_refusal(sharded, other);
        check(!reason.empty(),
              "rank " + std::to_string(other.rank) + " of world " +
                  std::to_string(other.world_size) + " on axis " +
                  std::string(ninfer::multi::axis_name(other.axis)) +
                  " must refuse a record it did not write");
    }

    // THE AXIS IS NOT IN THE RECORD, AND THAT IS A CHECKED LIMITATION RATHER THAN AN OVERSIGHT.
    // The shard axis costs 8 bytes and the release reason's word costs 4, so the record's honest
    // pool is 4 bytes -- smaller than one descriptor (8), i.e.
    // `recall_record_free_bytes_honest() == 4 < kRecallRegionDescriptorBytes`, and
    // `recall_extra_regions_after_shard_axis() == 0`. There is therefore no whole word left for
    // an axis field even if the pool were word-granular. A (rank, world) pair does not say WHICH
    // axis produced it, and a pipeline world with the same rank count accepts a tensor record of
    // the same (rank, world).
    //
    // That is tolerable because the axis is a RUN-level property, not a per-record one: a process
    // runs one world with one axis, `sharded_name_acceptance()` (core/shard_plan.h) DOES carry the
    // axis in the artifact name, and `world_shape_refusal()` refuses a world that does not name
    // its axis at all. So the record's job is to catch the "different rank or different rank
    // count" case, and the axis is caught one level up. This assertion exists so the limitation
    // is a recorded, tested fact: if a future change wants records from two axes to coexist in
    // one directory, the ledger above has to grow and this check must go red first.
    RecallRecord tensor_record = fresh;
    stamp_recall_record_shard(tensor_record, tp(2, 1));
    check(recall_record_shard_refusal(tensor_record, WorldShape{2, 1, ParallelAxis::Pipeline})
              .empty(),
          "the same (rank, world) on a different axis is ACCEPTED by the record's rule: the "
          "8-byte shard axis cannot carry the axis, and the artifact-name rule "
          "(sharded_name_acceptance) is where the axis is actually enforced");
    // Red control for that limitation, which is also what keeps the check above from being a
    // tautology: change the rank and the very same function refuses.
    check(!recall_record_shard_refusal(tensor_record, tp(2, 0)).empty(),
          "the axis-blindness must be confined to the axis: the same record read by a different "
          "rank of the same world must still be refused");

    // Malformed records: a reader must refuse rather than mis-read, in all four shapes.
    RecallRecord zero_world = fresh;
    zero_world.shard_world   = 0;
    check(!recall_record_shard_refusal(zero_world, single).empty(), "world 0 must be refused");
    RecallRecord out_of_range = fresh;
    out_of_range.shard_rank   = 3;
    out_of_range.shard_world  = 2;
    check(!recall_record_shard_refusal(out_of_range, tp(2, 0)).empty(),
          "rank >= world must be refused");
    RecallRecord bad_run = fresh;
    bad_run.shard_world  = 0;
    check(!recall_record_shard_refusal(bad_run, WorldShape{0, 0, ParallelAxis::None}).empty(),
          "a malformed reader world must be refused before the record is even looked at");

    // The red control: the refusal must be ABLE to return empty for a mismatched record, i.e. it
    // must be a comparison and not a constant. Both directions are asserted so neither "refuse
    // everything" nor "accept everything" survives.
    check(recall_record_shard_refusal(fresh, single).empty() &&
              !recall_record_shard_refusal(sharded, single).empty(),
          "the same function must return empty and non-empty for two records that differ ONLY "
          "in the shard words -- otherwise the check is a constant");
}

// ---------------------------------------------------------------------------
// D. THE ATTENTION COMBINE -- the all-reduce's algebra, and the one no-op it must be
// ---------------------------------------------------------------------------
//
// A tensor world splits the KV heads, so each rank produces a PARTIAL softmax over its own heads.
// The whole row is the log-sum-exp merge of the partials. What is checkable on one GPU is the
// ALGEBRA and the world_size == 1 case; what is not checkable here is the cost of the transport,
// and the report says so.

// One rank's partial, built from the (score, value) pairs it can see. This is the honest
// construction: the partial is computed from real data, so the merge is tested against a
// reference computed the same way, not against a literal this file invented.
struct Partial {
    double m = -std::numeric_limits<double>::infinity();
    double l = 0.0;
    double o = 0.0;
};

[[nodiscard]] Partial make_partial(const std::vector<std::pair<double, double>>& visible) {
    if (visible.empty()) { return Partial{}; }
    double m = visible.front().first;
    for (const auto& kv : visible) { m = kv.first > m ? kv.first : m; }
    double l = 0.0, o = 0.0;
    for (const auto& kv : visible) {
        const double w = std::exp(kv.first - m);
        l += w;
        o += kv.second * w;
    }
    return Partial{m, l, o};
}

[[nodiscard]] AttentionCombine make_combine(const std::vector<std::pair<double, double>>& visible) {
    const Partial p = make_partial(visible);
    return AttentionCombine{p.m, p.l, p.o};
}

[[nodiscard]] double cells_at(const std::vector<std::pair<double, double>>& v) {
    return v.empty() ? 0.0 : make_partial(v).o / make_partial(v).l;
}

void test_d1_world_size_one_performs_no_collective_and_changes_no_bit() {
    // The property that regresses when someone wires a collective in unconditionally. It is
    // asserted as EXACT equality, not as a tolerance, because "the collective is the identity
    // here" is a claim about the code path, not about rounding.
    const AttentionCombine whole{1.25, 3.5, -0.75};
    const std::vector<AttentionCombine> one{whole};

    check(attention_combine_outcome(ParallelAxis::None) == CombineOutcome::None,
          "the single device must be outcome None: there is nothing to reduce over");
    check(attention_combine_outcome(ParallelAxis::Tensor) == CombineOutcome::MergeRequired,
          "the tensor axis splits the row, so a merge is REQUIRED");
    check(attention_combine_outcome(ParallelAxis::Pipeline) == CombineOutcome::Replicated,
          "the pipeline axis replicates attention: every stage computes whole rows");
    check(!attention_is_replicated(ParallelAxis::Tensor),
          "the tensor axis must not be reported as replicated");
    check(attention_is_replicated(ParallelAxis::Pipeline) &&
              attention_is_replicated(ParallelAxis::None),
          "both other axes replicate attention");

    bool performed = true;
    const AttentionCombine result = all_reduce_attention_combine(one, CombineOutcome::None,
                                                                &performed);
    check(!performed, "world_size 1 must report that NO collective was performed");
    check(result.m == whole.m && result.l == whole.l && result.o == whole.o,
          "and must return the value itself, bit for bit");

    // Red control: the "no collective" report must be able to be true. A world that needs a merge
    // and has two partials must set it.
    const std::vector<AttentionCombine> two{whole, whole};
    bool performed_two = false;
    (void)all_reduce_attention_combine(two, CombineOutcome::MergeRequired, &performed_two);
    check(performed_two, "a merge over two partials must report that a collective happened");
}

void test_d2_split_then_join_equals_the_whole_row() {
    // A row of 8 keys with distinct scores, partitioned across world_size ranks by KEY, which is
    // what a head split does to a row's visible set (each rank sees its own heads' keys).
    std::vector<std::pair<double, double>> all;
    for (int i = 0; i < 8; ++i) {
        all.emplace_back(0.5 * static_cast<double>(i) - 1.0, std::sin(0.7 * i));
    }
    const double reference = cells_at(all);

    for (const std::uint32_t world : {1U, 2U, 4U, 8U}) {
        // Each rank sees every world-th key, so the union is the whole row and no rank's set is a
        // prefix of another's -- the case a naive combine gets wrong.
        std::vector<AttentionCombine> partials;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            std::vector<std::pair<double, double>> mine;
            for (std::size_t i = rank; i < all.size(); i += world) { mine.push_back(all[i]); }
            partials.push_back(make_combine(mine));
        }
        const CombineOutcome outcome = attention_combine_outcome(ParallelAxis::Tensor);
        const AttentionCombine merged = all_reduce_attention_combine(partials, outcome);
        const double got = merged.l == 0.0 ? 0.0 : merged.o / merged.l;
        check(std::fabs(got - reference) <= 1e-12 * (1.0 + std::fabs(reference)),
              "world " + std::to_string(world) + ": split then join must equal the whole row "
              "(got " + std::to_string(got) + ", want " + std::to_string(reference) + ")");

        // Order independence: the merge is associative and commutative, which is what makes an
        // all-reduce legal at all. A reduction whose result depended on rank order would need a
        // barrier AND a fixed order, which is a different (and much more expensive) contract.
        //
        // THE CAVEAT, stated rather than glossed: `max` is exact, so `m` must agree BIT FOR BIT
        // in any order, while `l` and `o` are sums, and floating-point addition is not
        // associative. The tolerance below is a relative 1e-12, which is around a few hundred
        // ulps at these magnitudes; the naive-merge control further down is off by 2.7e-2, i.e.
        // ten orders of magnitude above it, so this tolerance cannot hide a real dependence.
        //
        // The consequence is worth recording, because it is a serving-engine requirement and not
        // a numerics footnote: the merge gives bit-identical results only for a FIXED reduction
        // order, so any implementation that wants a run to be reproducible must fix the order
        // (rank order, or a tree with a fixed shape) rather than rely on "it does not matter".
        std::vector<AttentionCombine> reversed(partials.rbegin(), partials.rend());
        const AttentionCombine other = all_reduce_attention_combine(reversed, outcome);
        check(other.m == merged.m,
              "world " + std::to_string(world) +
                  ": the row max is a max, so it must agree EXACTLY in any reduction order");
        const auto close = [](double x, double y) {
            return std::fabs(x - y) <= 1e-12 * (1.0 + std::fabs(x) + std::fabs(y));
        };
        check(close(other.l, merged.l) && close(other.o, merged.o),
              "world " + std::to_string(world) + ": the merge must be order-independent up to "
              "floating-point association (got l=" + std::to_string(other.l) + "/" +
                  std::to_string(merged.l) + ", o=" + std::to_string(other.o) + "/" +
                  std::to_string(merged.o) + ")");
    }

    // THE RED CONTROL, and it is the reason this group is worth anything: a combine that skipped
    // the max-rescale (i.e. a plain sum of numerators and denominators) must produce a DIFFERENT
    // answer on these partials, so the agreement above is not forced by the inputs being
    // trivially mergeable.
    std::vector<AttentionCombine> partials;
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        std::vector<std::pair<double, double>> mine;
        for (std::size_t i = rank; i < all.size(); i += 2) { mine.push_back(all[i]); }
        partials.push_back(make_combine(mine));
    }
    const double naive = (partials[0].o + partials[1].o) / (partials[0].l + partials[1].l);
    check(std::fabs(naive - reference) > 1e-9,
          "the naive sum-of-partials must be WRONG on these inputs (" + std::to_string(naive) +
              " vs " + std::to_string(reference) +
              "), which is what makes the merge test non-vacuous");

    // And the merge must be the identity on ONE partial, whatever the outcome says.
    for (const auto outcome : {CombineOutcome::None, CombineOutcome::Replicated,
                               CombineOutcome::MergeRequired}) {
        const AttentionCombine only = all_reduce_attention_combine({AttentionCombine{2.0, 4.0, 1.0}},
                                                                   outcome);
        check(only.m == 2.0 && only.l == 4.0 && only.o == 1.0,
              "one partial must merge to itself under every outcome");
    }
}

void test_d3_an_empty_rank_contributes_nothing_and_makes_no_nan() {
    // A rank whose visible set is empty contributes (m=-inf, l=0, o=0), and merging it must not
    // produce a NaN through exp(-inf - -inf). This case is not hypothetical: it is what a rank
    // with no keys in a narrowed window looks like.
    const std::vector<std::pair<double, double>> all{{-1.0, 0.5}, {0.25, -0.25}, {1.5, 0.75}};
    const double reference = cells_at(all);

    const AttentionCombine full  = make_combine(all);
    const AttentionCombine empty = make_combine({});
    const AttentionCombine merged = combine_merge(full, empty);
    check(std::isfinite(merged.m) && std::isfinite(merged.l) && std::isfinite(merged.o),
          "merging an empty partial must not produce a non-finite value");
    check(merged.l == full.l && merged.o == full.o,
          "an empty partial must be the identity: l and o unchanged");
    const double got = merged.o / merged.l;
    check(std::fabs(got - reference) <= 1e-12 * (1.0 + std::fabs(reference)),
          "and the merged row must equal the whole row");
    const AttentionCombine both = combine_merge(empty, empty);
    check(both.l == 0.0, "two empty partials must give an empty result, not a NaN");
    check(attention_combine_outcome(ParallelAxis::Tensor) == CombineOutcome::MergeRequired &&
              all_reduce_attention_combine({}, CombineOutcome::MergeRequired).l == 0.0,
          "an all-reduce over no ranks must be empty rather than undefined");
}

// ---------------------------------------------------------------------------
// E. ONE PROCESS, N RANKS -- the multi-rank configuration, EXERCISED
// ---------------------------------------------------------------------------
//
// Everything above tests one structure at a time. This group is the simulator proper: it builds
// the WHOLE rank context for every rank of a world, in one process, the way
// core/virtual_device.h says the ranks should be built, and then asserts the properties that
// only exist BETWEEN ranks:
//
//   * every rank gets a different KV-head slice and a different weight-column slice;
//   * every rank's spill cell and spill SLOT are different from every other's;
//   * the world's spill spend stays inside the world's cap;
//   * every rank's 64-byte record is accepted by that rank and refused by every other;
//   * the attention combine needs a merge for world > 1 and is the exact identity for world 1;
//   * a world whose rank count does not divide the model's KV heads is REFUSED, not rounded.
//
// What it does NOT and cannot establish, and the report says so in the same words: nothing here
// is a measurement. The ranks share one HBM and time-slice one SM pool, so communication cost,
// bandwidth-bound behaviour and per-rank wall-clock scaling are absent rather than noisy. All
// three are `pending hardware`.

// A rank's whole context, exactly as core/virtual_device.h's binding describes it plus the
// storage arithmetic this header owns. Nothing here is a global: this struct IS the rank.
struct RankContext {
    WorldShape world{};
    ShardPlan  shard{};
    RankPageStride stride{};
    SpillSlotSpace slots{};
    std::string spill_cell;
    // The engine's own cold-slot stride for one layer, divided by the rank's share of the heads.
    std::size_t per_layer_cold_slot_bytes = 0;
};

// The model geometry's shard-relevant part. The numbers are qwen3_6's published facts where they
// are known -- 16 full-attention layers, 8 KV heads, head_dim 128, and a cold slot stride that
// covers a whole (page, head, plane) record -- and the query-head count is derived
//
// NOTE ON kColdSlot BELOW, because it is 6688 and the tree's cold slot stride is 9632 B
// (product/kv_bit_budget.h kKvBitBudgetColdSlotBytes, since 2026-09-18): the literal is a
// STAND-IN and is DELIBERATELY NOT re-pointed at the header. plan_spill_slot_space() below
// divides a 4 GiB cap by this stride and the group's checks compare the resulting per-rank
// byte counts, so moving the stride moves the quantities under assertion -- that would be a
// change to what this test measures, not a stale-number fix.
//
// from the split that is legal, so the group is a test of the ARITHMETIC rather than of a
// remembered config.
[[nodiscard]] bool build_world(std::uint32_t world_size, std::uint32_t weight_columns,
                               std::vector<RankContext>& out) {
    out.clear();
    constexpr std::uint32_t kKvHeads    = 8;
    constexpr std::uint32_t kHeadDim    = 128;
    constexpr std::uint32_t kTextLayers = 16;
    constexpr std::size_t   kColdSlot   = 6688; // STAND-IN, not the tree's stride (9632 B):
                                               // the checks below are value-dependent, so do
                                               // NOT re-point this at kKvBitBudgetColdSlotBytes
    constexpr std::uint64_t kSpillCap   = 4ULL * 1024 * 1024 * 1024; // --cold-disk-bytes

    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        RankContext context;
        // A world of 1 IS the identity world: `axis=tp` with `world_size 1` is refused by
        // shard_plan.h's own world_shape_refusal, which is the point of that rule -- a
        // single-device world must say axis=none so the KV granularity contract can be checked.
        context.world = world_size == 1U ? WorldShape{1, 0, ParallelAxis::None}
                                         : tp(world_size, rank);
        context.shard = ninfer::multi::plan_shards(
            context.world, ninfer::multi::ModelGeometry{kTextLayers, 2U * kKvHeads, kKvHeads,
                                                        kHeadDim, weight_columns});
        if (!context.shard.ok) { return false; }

        // The rank's page record: the whole page's cold-slot stride divided by its share of the
        // heads. `rank_page_stride` refuses a share that does not divide exactly.
        const std::size_t whole = kColdSlot * kKvHeads;
        context.stride = rank_page_stride(context.world, whole);
        if (!context.stride.ok) { return false; }
        context.per_layer_cold_slot_bytes = context.stride.per_rank_stride;

        context.slots = plan_spill_slot_space(context.world, kSpillCap,
                                             context.per_layer_cold_slot_bytes,
                                             BudgetScope::WorldTotal);
        if (!context.slots.ok) { return false; }
        context.spill_cell = ninfer::multi::cold_spill_cell_path("/tmp", /*layer=*/0, context.world);
        out.push_back(std::move(context));
    }
    return true;
}

void test_e1_the_whole_world_is_built_per_rank_in_one_process() {
    // weight_columns = (q_heads + 2*kv_heads) * head_dim with q_heads = 2*kv_heads here, i.e.
    // 24 kv-groups worth of N. 24 divides 2, 3, 4, 6, 8, 12, 24 so all the worlds below are legal.
    constexpr std::uint32_t kWeightColumns = 24U * 128U;
    for (const std::uint32_t world : {1U, 2U, 4U, 8U}) {
        std::vector<RankContext> ranks;
        const bool built = build_world(world == 1U ? 1U : world, kWeightColumns, ranks);
        check(built, "world " + std::to_string(world) + " must build all of its rank contexts");
        if (!built) { continue; }
        check(ranks.size() == world, "one context per rank, in this one process");
        if (world == 1U) { ranks[0].world = WorldShape{1, 0, ParallelAxis::None}; }

        std::set<std::int32_t> kv_head_begins, column_begins, world_slots;
        std::set<std::string> cells;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const RankContext& c = ranks[rank];
            check(kv_head_begins.insert(c.shard.kv_heads.begin).second,
                  "rank " + std::to_string(rank) + " must own a KV-head slice no other rank owns");
            check(column_begins.insert(c.shard.weight_columns.begin).second,
                  "rank " + std::to_string(rank) +
                      " must own a weight-column block no other rank owns");
            check(c.shard.text_layers.begin == 0U && c.shard.text_layers.end == 16U,
                  "the TENSOR axis must leave every text layer whole on every rank");
            check(cells.insert(c.spill_cell).second,
                  "rank " + std::to_string(rank) + "'s spill cell must be its own: " +
                      c.spill_cell);

            // The slot each rank's allocator hands out first, through the partition.
            PerProcessAllocator local(static_cast<std::size_t>(c.slots.slots_per_rank));
            const std::int32_t slot = c.slots.world_slot(local.allocate());
            check(slot >= 0 && c.slots.range.contains(slot),
                  "rank " + std::to_string(rank) + "'s first slot must be inside its own share");
            check(world_slots.insert(slot).second,
                  "rank " + std::to_string(rank) +
                      "'s first slot must differ from every other rank's -- this is the exact "
                      "collision MULTIDEV-X measured, negated");

            // The record: written by this rank, accepted by this rank, refused by all others.
            using ninfer::spec::turn_recall::RecallRecord;
            RecallRecord record{};
            stamp_recall_record_shard(record, c.world);
            check(recall_record_shard_refusal(record, c.world).empty(),
                  "rank " + std::to_string(rank) + " must accept its own record");
            for (std::uint32_t other = 0; other < world; ++other) {
                if (other == rank) { continue; }
                check(!recall_record_shard_refusal(record, ranks[other].world).empty(),
                      "rank " + std::to_string(other) + " must refuse rank " +
                          std::to_string(rank) + "'s record");
            }
        }
        check(cells.size() == world, "the world's spill cells must be pairwise distinct");
        check(world_slots.size() == std::min<std::size_t>(world, 1U) * (world == 1U ? 1U : world),
              "the world's first slots must be pairwise distinct");

        // The world's spend stays inside the world's cap: slots_per_rank x world x per-rank stride.
        if (world > 1U) {
            const std::uint64_t spend =
                static_cast<std::uint64_t>(ranks[0].slots.slots_per_rank) *
                ranks[0].per_layer_cold_slot_bytes * world;
            check(spend <= 4ULL * 1024 * 1024 * 1024,
                  "world " + std::to_string(world) + " must not spend more than --cold-disk-bytes "
                  "(spend " + std::to_string(spend) + " of 4294967296)");
        }
    }
}

void test_e2_the_virtual_binding_is_what_names_the_rank() {
    // The rank context above is built from a hand-made WorldShape. This group checks that the
    // SIMULATOR produces that same world, so the two are one mechanism and not two.
    using ninfer::multi::validate_virtual_request;
    using ninfer::multi::VirtualProfile;
    using ninfer::multi::VirtualRequest;

    constexpr std::uint32_t kWeightColumns = 24U * 128U;
    for (const std::uint32_t world : {2U, 4U}) {
        std::vector<RankContext> ranks;
        check(build_world(world, kWeightColumns, ranks), "the world must build");

        for (std::uint32_t rank = 0; rank < world; ++rank) {
            VirtualRequest request;
            request.requested        = true;
            request.acknowledged     = true;
            request.world_size       = world;
            request.rank             = rank;
            request.axis             = ParallelAxis::Tensor;
            request.profile          = VirtualProfile::V100;
            request.world_size_raw   = std::to_string(world);
            request.profile_name_raw = "v100";
            // 170 SMs and 32 GiB: the facts of this box, passed in so the check stays host-only.
            const auto binding = validate_virtual_request(request, 1, 170, 32ULL * 1024 * 1024 * 1024);
            check(binding.active, "the virtual binding must be honoured for rank " +
                                      std::to_string(rank) + ": " + binding.refusal_detail);
            if (!binding.active) { continue; }
            check(binding.world == ranks[rank].world,
                  "the simulator's world for rank " + std::to_string(rank) +
                  " must be the world the rank contexts were built from");
            // The two mechanisms must agree about the rank's memory too: the virtual card's 16 GiB
            // is what a per-rank residency decision would be taken against, and it is a BUDGET
            // shared with the other ranks -- never a partition.
            check(binding.memory_budget_bytes == 16ULL * 1024 * 1024 * 1024,
                  "a virtual v100 rank must be told it has 16 GiB");
            check(binding.sm_budget == 170U / world,
                  "and its SM budget must be the physical count divided by the world");
        }
    }
}

void test_e3_a_world_that_does_not_divide_is_refused() {
    // The refusal that keeps the whole thing honest: 8 KV heads cannot be split 3 ways, and a
    // world that says otherwise must be refused rather than quietly rounded to a 3-head rank.
    constexpr std::uint32_t kWeightColumns = 24U * 128U;
    std::vector<RankContext> ranks;
    check(!build_world(3, kWeightColumns, ranks),
          "world 3 against 8 KV heads must be refused: the split would cut a GQA group across "
          "ranks, so a rank's query heads would attend to kv heads it does not hold");
    check(ranks.empty(), "and a refused world must build NO rank context at all");

    // The refusal must name the arithmetic, not just fail.
    const auto plan = ninfer::multi::plan_shards(
        tp(3, 0), ninfer::multi::ModelGeometry{16, 16, 8, 128, kWeightColumns});
    check(!plan.ok, "the shard plan itself must refuse world 3 against 8 KV heads");
    check(plan.reason.find("kv_heads") != std::string::npos &&
              plan.reason.find("divisible") != std::string::npos,
          "and the refusal must name the field and the rule, got: " + plan.reason);

    // Red control: raise the rank count to a divisor and the same call must succeed, so the
    // refusal above is about divisibility and not about the world size being unusual.
    const auto ok_plan = ninfer::multi::plan_shards(
        tp(4, 0), ninfer::multi::ModelGeometry{16, 16, 8, 128, kWeightColumns});
    check(ok_plan.ok, "world 4 must be accepted: " + ok_plan.reason);
}

// ---------------------------------------------------------------------------
// E4. BALANCE -- the part of "does it scale" that is measurable on ONE device
// ---------------------------------------------------------------------------
//
// Aggregate throughput scaling with rank count is not measurable on a single GPU, and the
// transport is not this engine's component. BALANCE is: it is a property of the shard plan, it
// needs no device at all, and a plan that hands one rank twice another's work is a
// shard-arithmetic defect that a virtual world can then CONFIRM against per-rank wall clock.
void test_e4_the_shard_plan_is_balanced() {
    constexpr std::uint32_t kWeightColumns = 24U * 128U;
    constexpr std::uint32_t kKvHeads       = 8;
    constexpr std::uint32_t kTextLayers    = 16;

    // The tensor axis must be PERFECTLY balanced, and the ratio is asserted as an equality
    // because both of its divisions are exact by construction: shard_plan.h refuses a
    // non-divisible split rather than rounding it, so a 1.5x imbalance cannot exist on this axis.
    for (const std::uint32_t world : {1U, 2U, 4U, 8U}) {
        std::vector<ShardPlan> plans;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            plans.push_back(ninfer::multi::plan_shards(
                world == 1U ? WorldShape{1, 0, ParallelAxis::None} : tp(world, rank),
                ninfer::multi::ModelGeometry{kTextLayers, 2U * kKvHeads, kKvHeads, 128,
                                             kWeightColumns}));
        }
        const auto balance = shard_balance(plans);
        check(balance.ok, "world " + std::to_string(world) + "'s balance must be computable: " +
                              balance.reason);
        check(balance.imbalance == 1.0,
              "the tensor axis must be PERFECTLY balanced at world " + std::to_string(world) +
                  ", got " + std::to_string(balance.imbalance) + "x");
        check(balance.min_work == balance.max_work,
              "and min == max work units, because both divisions are exact");
        // The work units must actually divide: a "balanced" world in which every rank got ZERO
        // work would also have a ratio of 1.0, so the magnitude is asserted too.
        check(balance.max_work > 0 && balance.world_size == world,
              "and the balance must describe real work for every rank");
    }

    // The pipeline axis is ALLOWED to be uneven -- an uneven pipeline costs balance only, because
    // every stage's layers are complete within that stage. So the check is the one-sided bound,
    // not an equality: it must not be MORE than one layer's worth out of balance.
    for (const std::uint32_t world : {3U, 5U}) {
        std::vector<ShardPlan> plans;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            plans.push_back(ninfer::multi::plan_shards(
                WorldShape{world, rank, ParallelAxis::Pipeline},
                ninfer::multi::ModelGeometry{kTextLayers, 16, kKvHeads, 128, kWeightColumns}));
        }
        const auto balance = shard_balance(plans);
        check(balance.ok, "the pipeline world's balance must be computable: " + balance.reason);
        check(balance.max_work - balance.min_work <= 1U,
              "an uneven pipeline may differ by at most ONE layer, got " +
                  std::to_string(balance.max_work - balance.min_work));
        check(balance.imbalance <= 2.0,
              "and never by a factor; got " + std::to_string(balance.imbalance));
    }

    // Red control: the balance function must be able to report an IMBALANCE, or the equalities
    // above would hold for a function that returns 1.0 unconditionally.
    std::vector<ShardPlan> lopsided;
    lopsided.push_back(ninfer::multi::plan_shards(WorldShape{4, 0, ParallelAxis::Pipeline},
                                                 ninfer::multi::ModelGeometry{16, 16, kKvHeads,
                                                                              128, kWeightColumns}));
    // A hand-built second plan with a fifth of the work: same axis, fewer layers.
    ShardPlan starved = lopsided[0];
    starved.text_layers = ninfer::multi::Range{0, 3};
    lopsided.push_back(starved);
    const auto uneven = shard_balance(lopsided);
    check(uneven.ok && uneven.imbalance > 1.0,
          "a plan with a starved rank must report an imbalance > 1.0, got " +
              std::to_string(uneven.imbalance));
    // And the units must be the axis's own: the pipeline axis counts LAYERS, not columns.
    check(shard_work_units(lopsided[0]) == lopsided[0].text_layers.count(),
          "the pipeline axis's work unit must be its text layers");
    const ShardPlan tensor_plan = ninfer::multi::plan_shards(
        tp(2, 0), ninfer::multi::ModelGeometry{kTextLayers, 16, kKvHeads, 128, kWeightColumns});
    check(shard_work_units(tensor_plan) ==
              tensor_plan.weight_columns.count() + tensor_plan.kv_heads.count(),
          "the tensor axis's work unit must be its weight columns plus its KV heads");

    // Refusals rather than silent zeros: a plan-less world, a failed plan, and a mixed axis.
    check(!shard_balance({}).ok, "no ranks at all must be refused");
    std::vector<ShardPlan> failed;
    failed.push_back(ninfer::multi::refuse("constructed failure"));
    check(!shard_balance(failed).ok, "a rank with no plan must be refused");
    check(shard_balance(failed).reason.find("constructed failure") != std::string::npos,
          "and the refusal must carry the underlying reason");
    std::vector<ShardPlan> mixed;
    mixed.push_back(ninfer::multi::plan_shards(tp(2, 0),
                                              ninfer::multi::ModelGeometry{16, 16, kKvHeads, 128,
                                                                           kWeightColumns}));
    mixed.push_back(ninfer::multi::plan_shards(WorldShape{2, 1, ParallelAxis::Pipeline},
                                              ninfer::multi::ModelGeometry{16, 16, kKvHeads, 128,
                                                                           kWeightColumns}));
    check(!shard_balance(mixed).ok, "a world whose ranks disagree about the axis must be refused");
    check(shard_balance_line(uneven).find("imbalance") != std::string::npos,
          "the report line must carry the imbalance");
}

} // namespace

int main() {
    // A1 FIRST, and deliberately: it runs the UNFIXED allocator and requires it to collide, so
    // the A2 group's "no collision" is a comparison against a failing control rather than a
    // check that would pass on any input. It was left out of this list by accident once, and the
    // -Wunused-function warning is what caught it -- the group would otherwise have been dead
    // code that still counted as written.
    test_a1_the_unpartitioned_space_must_collide();
    test_e1_the_whole_world_is_built_per_rank_in_one_process();
    test_e2_the_virtual_binding_is_what_names_the_rank();
    test_e3_a_world_that_does_not_divide_is_refused();
    test_e4_the_shard_plan_is_balanced();
    test_a2_the_partitioned_space_must_not_collide();
    test_a3_the_accounting_is_divided_exactly_once();
    test_b1_the_page_stride_must_divide_exactly();
    test_b2_the_capacity_is_off_by_exactly_world_size_when_scopes_mix();
    test_c1_the_region_budget_ledger();
    test_c2_the_reader_must_refuse_another_worlds_record();
    test_d1_world_size_one_performs_no_collective_and_changes_no_bit();
    test_d2_split_then_join_equals_the_whole_row();
    test_d3_an_empty_rank_contributes_nothing_and_makes_no_nan();

    std::cout << "shard_rank_axis: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
