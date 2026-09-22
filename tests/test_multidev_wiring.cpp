// test_multidev_wiring.cpp -- the three wiring facts a multi-device world has to satisfy,
// each with the red control that shows the check is not vacuous.
//
// This file exists because the first version of this line's analysis was WRONG in its
// framing: it argued that "splitting KV by attention head breaks the page's atomicity".
// The page's eviction PREDICATE does not break (it depends only on replicated inputs), and
// the on-disk arithmetic breaks for a much more concrete reason. The three groups below are
// the concrete reasons:
//
//   W1  THE ARCH TABLE IS EXACT-MATCH, AND THE TOOLCHAIN IS NOT.
//       `nvcc --list-gpu-arch` on this host prints twelve compute capabilities; kArchLadder
//       had rows for ten of them. A card landing on an unlisted-but-buildable capability is
//       refused by a table lookup, which is "using the number instead of the evidence" --
//       the same defect the deleted `sm() != 120` had, wearing a table.
//
//   W2  THE 64-BYTE RECORD'S REGION BUDGET IS ARITHMETIC, NOT AN OPINION.
//       One page = one region = 8 bytes of descriptor, and the shard axis needs 8 of the 16
//       free bytes. What does not fit is a second region WITH its own codec.
//
//   W3  THE SPILL CELL HAS A LAYER AXIS AND NO RANK AXIS, AND THE ALLOCATOR IS PER-PROCESS.
//       Every rank deterministically opens the same path and takes the same first slot.
//
// Host-only: no CUDA header, no device, no GPU.

#include "core/arch_caps.h"
#include "core/shard_plan.h"
#include "spec/turn_recall_journal.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::caps::arch_rung;
using ninfer::multi::cold_spill_cell_path;
using ninfer::multi::ParallelAxis;
using ninfer::multi::WorldShape;
using ninfer::spec::turn_recall::kRecallRecordBytes;
using ninfer::spec::turn_recall::kRecallRegionDescriptorBytes;
using ninfer::spec::turn_recall::kRecallRecordFreeBytes;
using ninfer::spec::turn_recall::kRecallReleaseReasonBytes;
using ninfer::spec::turn_recall::kRecallShardAxisBytes;
using ninfer::spec::turn_recall::RecallRecord;
using ninfer::spec::turn_recall::recall_extra_regions_after_shard_axis;
using ninfer::spec::turn_recall::recall_record_free_bytes_honest;
using ninfer::spec::turn_recall::recall_record_free_bytes_raw_after_shard_axis;

int failures = 0;
int checks   = 0;
int reported = 0;
constexpr int kReportLimit = 25;

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

// ---------------------------------------------------------------------------
// W1. Every capability the toolchain can build must be handled WITHOUT a number-based
// refusal.
// ---------------------------------------------------------------------------
//
// The list below is the verbatim output of, on this host:
//     /usr/local/cuda/bin/nvcc --list-gpu-arch
//   -> compute_75 compute_80 compute_86 compute_87 compute_88 compute_89 compute_90
//      compute_100 compute_110 compute_103 compute_120 compute_121
// (identical for /usr/local/cuda-13.1/bin/nvcc and /usr/local/cuda-13.3/bin/nvcc).
// It is pinned here as a constant so this check is a HOST check with no compiler
// dependency, and so that a future toolkit whose list grows fails this test instead of
// silently refusing the new card.
constexpr int kToolchainArches[] = {75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121};

// The invariant is deliberately NOT "the ladder has a row for every buildable sm". That
// would force an unmeasured capability set into the table for a card nobody has measured --
// which is the same sin as the deleted `sm() != 120`, only better dressed. The invariant is
// the non-gate principle itself: for every buildable capability, EITHER the table has a row
// (so its floor is checkable) OR it is UnknownArch and the gate WARNS instead of refusing.
// What is forbidden is manufacturing a refusal out of a missing table row.
void test_every_buildable_arch_is_handled_without_a_number_refusal() {
    for (const int sm : kToolchainArches) {
        const auto report = ninfer::caps::evaluate_artifact_formats(
            sm, std::vector<NumericFormat>{NumericFormat::NVFP4});
        const bool has_row = arch_rung(sm) != nullptr;
        if (has_row) {
            check(report.verdict != ninfer::caps::Verdict::UnknownArch,
                  "sm_" + std::to_string(sm) + " has a ladder row but was reported as "
                  "UnknownArch");
            check(ninfer::caps::unknown_arch_warning(report, "x/y").empty(),
                  "a card with a ladder row must produce no unknown-arch warning");
            continue;
        }
        // No row: the ONLY acceptable handling is a warning, never a silent refusal.
        check(report.verdict == ninfer::caps::Verdict::UnknownArch,
              "sm_" + std::to_string(sm) +
                  " is buildable, has no ladder row, and was not reported as UnknownArch");
        const std::string warning = ninfer::caps::unknown_arch_warning(report, "m/w");
        check(!warning.empty(),
              "sm_" + std::to_string(sm) +
                  " is buildable with no ladder row: it MUST produce the warning that "
                  "replaces the refusal");
        check(warning.find(std::to_string(sm)) != std::string::npos,
              "the warning must name the capability it could not place");
        check(warning.find("kernel") != std::string::npos,
              "the warning must say where a refusal comes from instead (the route selector "
              "naming the missing kernel)");
    }
    // The specific shape, named so a regression says which one broke:
    //   87/88 -> Ampere GA10x refresh, same capability set as sm_86: ROWS ADDED (family
    //            evidence; equal-to-neighbour keeps the ladder monotone).
    //   110   -> no row was added ON PURPOSE: see the comment in the report. It is handled
    //            by the WARNING path. (My first attempt added a row with the sm_89 set and
    //            the tree's own test_arch_caps.cpp rejected it: "ladder capabilities are
    //            monotone at sm_110".)
    for (const int sm : {87, 88}) {
        check(arch_rung(sm) != nullptr,
              "sm_" + std::to_string(sm) + " is buildable and its capability set is the "
              "same Ampere GA10x family set as sm_86, so it must have a row");
        check(arch_rung(sm)->caps == arch_rung(86)->caps,
              "sm_" + std::to_string(sm) + "'s capability set must equal sm_86's; an EQUAL "
              "neighbour is what keeps the ladder's monotonicity invariant true");
    }
    check(arch_rung(110) == nullptr,
          "sm_110 must NOT acquire a row without a measured capability set; it is handled "
          "by the warning path");

    // THE GATE ITSELF, not just the warning text. This distinction is the whole fix:
    //   * a capability with NO ROW is "I cannot check", and the gate must let it through
    //     (the route selector is where it gets a conservative route, and where a shape with
    //     no kernel gets a refusal that NAMES that kernel);
    //   * a capability WITH A ROW whose floor the format misses is an EVIDENCE-BACKED gap,
    //     and THAT must still refuse.
    // Both directions are asserted, so neither "refuse everything unlisted" nor "warn about
    // everything" survives. (The first version of this test looked only at the warning text,
    // and the injection that PUT THE THROW BACK survived it -- a hole, now closed.)
    // The two directions, partitioned by whether the capability has a ROW. The first version
    // of this loop lumped {110, 87, 88, 999} together and asserted "no throw" for all four;
    // adding the sm_87/88 rows reddened it, because those two now DO have a set to check and
    // Ampere does not cover NVFP4's floor -- a correct refusal it was wrongly forbidding.
    //
    //   NO ROW                            -> must NOT refuse (the table cannot know)
    //   ROW, and the floor is not covered  -> MUST refuse (an evidence-backed gap)
    for (const int sm : {110, 999}) {
        bool threw = false;
        try {
            ninfer::caps::require_artifact_formats_supported(
                sm, std::vector<NumericFormat>{NumericFormat::NVFP4}, "m/w");
        } catch (const std::invalid_argument&) { threw = true; }
        check(!threw,
              "sm_" + std::to_string(sm) +
                  " has no capability set to check against, so the gate must NOT manufacture "
                  "a refusal out of a missing table row");
    }
    // ---------------------------------------------------------------------------------------
    // THE {70} / {87, 88} SPLIT -- re-derived 2026-09-18. The list above used to hold all three
    // in ONE group, and a SECOND FLOOR is what makes that grouping wrong.
    // ---------------------------------------------------------------------------------------
    // The premise is untouched and is still asserted: all three ARE in the ladder and none of
    // them covers kind::mxf4nvf4. What no longer follows is "therefore the gate must refuse".
    // `src/core/arch_caps.h`'s FormatRequirement gained a SECOND, LOWER floor
    // (`fallback_required`), and `fp16_fallback_executable()` -- four clauses, fail-closed on an
    // unmeasured rung -- decides whether a given rung gets to use it. The two answers it
    // produces are exactly what this pair of groups now pins:
    //
    //   sm_87 / sm_88 -- CLAUSE 4 FAILS. Their kQpnMmaRungs cell is EmulatedFp16Pipe (measured:
    //                    224 CALLs into ptxas's FFMA routines, HMMA.884 = 0), so the fallback
    //                    refuses, the verdict is Unsupported, and the floor miss stays a refusal.
    //                    The check is the author's own, unchanged -- but it now passes
    //                    `qpn_in_build = true` EXPLICITLY, so the refusal is attributed to the
    //                    LOWERING and not to the kernel being absent from the build.
    //   sm_70       -- ALL FOUR CLAUSES HOLD. Its kQpnMmaRungs cell is HardwareMma884 (measured:
    //                    1024 HMMA.884.F32.F32, 0 CALLs into the emulation routine, 0
    //                    HFMA2.MMA), NVFP4's row names Cap::Fp16Mma as its second floor with a
    //                    real kernel file behind it, and that kernel is in this build. The verdict
    //                    is Supported WITH a recorded FormatFallback -- the loud answer, not a
    //                    silent one.
    //
    // WHY PINNING ONLY THE OLD OUTCOME WAS A DEFECT: it asserted the ABSENCE of the mechanism
    // rather than the mechanism, so it could only ever be satisfied by the tree NOT having a
    // second floor. Both directions are therefore asserted for sm_70, through the `qpn_in_build`
    // PARAMETER that `src/core/arch_caps.h:815-818` says exists for precisely this ("it lets one
    // test binary exercise the build-with-QPN and build-without-QPN worlds without a rebuild,
    // which is how the fail-closed half gets a red control instead of an assertion about a binary
    // nobody can produce"). That also takes this check out of dependence on whether THIS
    // translation unit received -DNINFER_HAVE_QPN -- a build-system fact, and not the subject of W1.
    for (const int sm : {87, 88}) {
        bool threw = false;
        try {
            // Both ARE in the ladder and neither covers kind::mxf4nvf4, so each is a real,
            // evidence-backed gap and the gate must still refuse it loudly -- in a build that HAS
            // the fallback kernel, which is what the explicit `true` makes the check say.
            ninfer::caps::require_artifact_formats_supported(
                sm, std::vector<NumericFormat>{NumericFormat::NVFP4}, "m/w",
                /*qpn_in_build=*/true);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw,
              "an EVIDENCE-backed floor miss (sm_" + std::to_string(sm) +
                  " has no kind::mxf4nvf4) must still be a refusal: the non-gate principle "
                  "removes number-based refusals, not real ones");
    }

    // sm_70: the same floor miss, and the two worlds the `qpn_in_build` parameter separates.
    {
        const int volta = 70;

        // (a) THE RED CONTROL, kept: with no fp16 fallback kernel in the build, the floor miss is
        // a refusal. This is the sentence the original loop was written to say, now made
        // independent of this TU's own -D so it cannot be turned on or off by a flags change.
        bool threw_without_fallback = false;
        try {
            ninfer::caps::require_artifact_formats_supported(
                volta, std::vector<NumericFormat>{NumericFormat::NVFP4}, "m/w",
                /*qpn_in_build=*/false);
        } catch (const std::invalid_argument&) { threw_without_fallback = true; }
        check(threw_without_fallback,
              "sm_70 has no kind::mxf4nvf4 and with NO fp16 fallback kernel in the build that "
              "floor miss must STILL be a refusal: the non-gate principle removes number-based "
              "refusals, not real ones");

        const auto volta_bare = ninfer::caps::evaluate_artifact_formats(
            volta, std::vector<NumericFormat>{NumericFormat::NVFP4}, /*qpn_in_build=*/false);
        check(volta_bare.verdict == ninfer::caps::Verdict::Unsupported &&
                  volta_bare.fallbacks.empty(),
              "and the report must agree with that refusal: without the fallback kernel sm_70 is "
              "Unsupported -- not UnknownArch, and not Supported");

        // (b) THE WORLD THIS BUILD IS IN. Same rung, same format, one build fact different.
        const auto volta_served = ninfer::caps::evaluate_artifact_formats(
            volta, std::vector<NumericFormat>{NumericFormat::NVFP4}, /*qpn_in_build=*/true);
        check(volta_served.verdict == ninfer::caps::Verdict::Supported,
              "with the QPN fp16 fallback kernel IN this build, sm_70's NVFP4 artifact IS "
              "supported -- by the SECOND floor, and the verdict says so");
        check(volta_served.fallbacks.size() == 1U,
              "and the fallback must be RECORDED rather than silently applied: exactly one "
              "FormatFallback, so a caller can print what carried the artifact");
        if (volta_served.fallbacks.size() == 1U) {
            const ninfer::caps::FormatFallback& fb = volta_served.fallbacks.front();
            check(fb.format == NumericFormat::NVFP4,
                  "the recorded fallback must be for the format that took it");
            check(fb.primary_missing == ninfer::caps::Cap::Mxf4Nvfp4BlockScale,
                  "it must name the floor sm_70 does NOT cover (kind::mxf4nvf4) -- a verdict of "
                  "Supported means less, not more, when the operator is not told which floor was "
                  "missed");
            check(fb.fallback_used == ninfer::caps::Cap::Fp16Mma,
                  "and the lower floor it DOES cover (Cap::Fp16Mma): the second floor is the whole "
                  "mechanism and Cap::Fp16Mma used to be a bit that decided nothing");
            check(fb.fallback_kernel.find("qpn") != std::string_view::npos,
                  "and the kernel that makes the claim TRUE must be named, so it can be checked "
                  "against a file in the tree instead of believed");
        }
        const std::string notice = ninfer::caps::render_fallback_notice(volta_served, "m/w");
        check(!notice.empty() && notice.find("FALLBACK") != std::string::npos,
              "a fallback-served artifact must produce the LOUD notice: a silent fallback would be "
              "indistinguishable from a card that meets the floor");
        check(notice.find("does NOT meet its own floor") != std::string::npos,
              "and the notice must say the floor was MISSED, so nobody is told sm_70 meets an fp4 "
              "floor that it does not have");

        // (c) THE FAIL-CLOSED CONTROL, and the reason (b) is not a blanket loosening. sm_80 is the
        // first rung whose kQpnMmaRungs cell is EmulatedFp16Pipe -- the mma ASSEMBLES there and
        // ptxas answers it with a software routine. No build fact may turn that into a route:
        // naming an fp16 tensor-core path and delivering an FMA-pipe simulation is the false
        // positive this second floor must never make.
        for (const int sm : {80, 86, 87, 88, 89, 90, 100, 103}) {
            check(!ninfer::caps::fp16_fallback_executable(
                      sm, NumericFormat::NVFP4, /*qpn_in_build=*/true),
                  "sm_" + std::to_string(sm) +
                      " lowers the fallback's mma in SOFTWARE, so the fallback must NOT be "
                      "executable there even though its kernel is in the build");
        }
    }

    // A genuinely unknown capability still warns, with the number in it.
    const auto unknown = ninfer::caps::evaluate_artifact_formats(
        999, std::vector<NumericFormat>{NumericFormat::NVFP4});
    check(unknown.verdict == ninfer::caps::Verdict::UnknownArch, "sm_999 must be UnknownArch");
    check(ninfer::caps::unknown_arch_warning(unknown, "m/w").find("999") != std::string::npos,
          "the warning must name the capability");
}

// The red control for W1: the coverage question must be able to FAIL. Re-derived from the
// pinned toolchain list against a hand-built table, so the check cannot pass merely because
// kArchLadder happens to be complete today.
void test_w1_control_detects_a_missing_row() {
    const std::set<int> good{75, 80, 86, 87, 88, 89, 90, 100, 103, 120, 121};  // 110 absent OK
    const std::set<int> bad{75, 80, 86, 89, 90, 100, 103, 120, 121};           // 87/88 gone
    auto uncovered = [](const std::set<int>& table) {
        std::vector<int> missing;
        for (const int sm : kToolchainArches) {
            if (table.find(sm) == table.end()) { missing.push_back(sm); }
        }
        return missing;
    };
    check(uncovered(good).size() == 1U && uncovered(good)[0] == 110,
          "the current table must leave exactly sm_110 uncovered (and that is the honest "
          "state: no measured set exists for it)");
    const std::vector<int> missing = uncovered(bad);
    check(missing == std::vector<int>({87, 88, 110}),
          "the pre-fix table must report exactly 87/88/110 -- this is what 'using the "
          "number instead of the evidence' looks like in a table");
}

// ---------------------------------------------------------------------------
// W2. The record's region budget.
// ---------------------------------------------------------------------------

void test_record_region_budget() {
    check(sizeof(RecallRecord) == kRecallRecordBytes,
          "the record must stay one fixed 64-byte unit after naming the shard words");
    check(kRecallRecordBytes == 64U, "the stride is the log's only format guarantee");
    check(kRecallRegionDescriptorBytes == sizeof(std::uint32_t) + sizeof(std::int32_t),
          "a region descriptor is layer_bytes + file_slot");

    // The shard axis costs 8 of the 16 free bytes and must leave at least one region of
    // room; and it must NOT leave two, because two regions would need two codec bytes.
    check(kRecallRecordFreeBytes == 16U, "the record's free words must be 4 x u32");
    check(kRecallShardAxisBytes == 8U, "the shard axis must be rank + world");
    check(recall_extra_regions_after_shard_axis() == 0U,
          "after the shard axis NO extra region descriptor fits; got " +
              std::to_string(recall_extra_regions_after_shard_axis()));
    // EVERY BYTE ACCOUNTED FOR, and this is the ONE AUTHORITY's identity rather than a
    // repeated arithmetic. The previous form of this check read
    //   kRecallRecordFreeBytes - kRecallShardAxisBytes - 1 * kRecallRegionDescriptorBytes == 0
    // i.e. "R=2 fits with ZERO slack" -- which counted `reserved`@28 as free. It is the
    // RELEASE REASON: recall_release_reason() reads `reserved & 0xFF`,
    // recall_set_release_reason() writes the WHOLE word, and the writer is live at
    // program_impl.h:11351. A second descriptor at @28 would have its stride clobbered by
    // the next release, so the word is occupied even though 3 of its 4 bytes read as zero.
    check(kRecallRecordFreeBytes == kRecallReleaseReasonBytes + kRecallShardAxisBytes +
                                        recall_record_free_bytes_honest(),
          "the ledger must account for every free word exactly ONCE: 16 == 4 (release "
          "reason) + 8 (shard axis) + 4 (honest pool)");
    check(recall_record_free_bytes_honest() < kRecallRegionDescriptorBytes,
          "and the reason the extra-region count is 0 is this inequality, not a coincidence");
    check(recall_record_free_bytes_raw_after_shard_axis() >
              recall_record_free_bytes_honest(),
          "the RAW subtraction must exceed the honest one: the word it miscounts is a live "
          "writer's field, not slack");

    // The defaults are the identity world, so every record already on disk reads back as
    // the world it was written in.
    const RecallRecord fresh{};
    check(fresh.shard_rank == 0U && fresh.shard_world == 1U,
          "(0,1) must be the default shard -- every pre-existing record was written by a "
          "single-device world");

    // THE CORRECTION, 2026-09-18. This group used to end by asserting that the
    // single-stride invariant, "not the byte count", is what refuses R>=2 -- and it read
    // the Mixed refusal as that invariant. The second half is false: both cold IO legs are
    // PER LAYER (program_impl.h:11814-11833 / :12559-12579) and the restore's codec comes
    // from `view.dtype`, so a page with two strides needs ONE region descriptor and Mixed
    // is admitted. What is true, and what this group still exists to pin, is that the
    // BYTE COUNT refuses R>=2 *with a per-region codec* -- the control below.
    check(ninfer::spec::turn_recall::recall_codec_admitted(
              ninfer::spec::turn_recall::RecallCodec::Mixed),
          "Mixed IS admitted: one region descriptor covers a page whose layers carry two "
          "strides, because each layer is addressed at its own extent in its own file");
    check(recall_extra_regions_after_shard_axis() == 0U,
          "and Mixed does not want a second region: the budget above is unchanged");
}

// The red control for W2: the ledger must be able to FAIL. If a second region with its own
// codec could fit, this arithmetic would say so.
void test_w2_control_the_ledger_can_fail() {
    // Two regions PLUS a per-region codec: 2 descriptors + 1 extra codec byte.
    const std::uint32_t needs = 2U * kRecallRegionDescriptorBytes + 1U;
    const std::uint32_t has   = recall_record_free_bytes_honest(); // the ONE AUTHORITY
    check(needs > has,
          "R=2 with a per-region codec must NOT fit in the free bytes after the shard "
          "axis (needs " + std::to_string(needs) + ", has " + std::to_string(has) + ")");
    // And the record's SINGLE region must still be legal, which it is for a reason that is
    // NOT "there is room for it in the free words": the primary region's two fields are
    // NAMED (@12 layer_bytes, @16 file_slot), so they cost nothing from the pool at all.
    // That is why the record is legal even with an honest pool smaller than a descriptor.
    check(recall_record_free_bytes_honest() < kRecallRegionDescriptorBytes &&
              kRecallRecordFreeBytes >= kRecallRegionDescriptorBytes,
          "the primary region is a NAMED field pair, not free-pool space -- which is why "
          "the record is legal with a pool smaller than one descriptor");
}

// ---------------------------------------------------------------------------
// W3. The spill cell's rank axis.
// ---------------------------------------------------------------------------

void test_spill_cell_needs_a_rank_axis() {
    const WorldShape single{1, 0, ParallelAxis::None};

    // Byte-identical to the tree's current expression for a single-device world.
    const std::string single_path = cold_spill_cell_path("/tmp", 0, single);
    check(single_path == "/tmp/ninfer_cold_L0.slot",
          "the single-device path must be byte-identical to the tree's current name "
          "(dir + \"/ninfer_cold_L\" + layer + \".slot\"), got " + single_path);
    check(cold_spill_cell_path("/tmp", 7, single) == "/tmp/ninfer_cold_L7.slot",
          "and for every layer");

    // THE DEFECT: without a rank axis every rank produces the SAME name.
    std::set<std::string> unsharded;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        // rank is deliberately ignored -- this reproduces the tree's expression exactly.
        (void)rank;
        unsharded.insert("/tmp/ninfer_cold_L0.slot");
    }
    check(unsharded.size() == 1U,
          "the tree's current expression collapses every rank onto one path; the "
          "per-process first-fit allocator then hands every rank slot 0, so the writes "
          "COLLIDE rather than interleave");

    // THE FIX: one distinct path per rank, and every path still layer-distinct.
    for (const std::uint32_t world : {2U, 4U, 8U}) {
        std::set<std::string> paths;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const WorldShape tp{world, rank, ParallelAxis::Tensor};
            for (std::uint32_t layer = 0; layer < 8; ++layer) {
                const std::string path = cold_spill_cell_path("/tmp", layer, tp);
                check(paths.insert(path).second,
                      "two (rank, layer) cells produced the same spill path: " + path);
                check(path.find("L" + std::to_string(layer) + ".r") != std::string::npos,
                      "the path must keep BOTH axes, got " + path);
            }
        }
        check(paths.size() == static_cast<std::size_t>(world) * 8U,
              "world " + std::to_string(world) + " x 8 layers must give " +
                  std::to_string(world * 8U) + " distinct paths, got " +
                  std::to_string(paths.size()));
    }
    const WorldShape tp1{4, 1, ParallelAxis::Tensor};
    check(cold_spill_cell_path("/tmp", 0, tp1) == "/tmp/ninfer_cold_L0.r1of4.slot",
          "the tp path must spell the shard, got " + cold_spill_cell_path("/tmp", 0, tp1));
    // A pipeline world has the same problem and gets the same treatment.
    const WorldShape pp2{2, 1, ParallelAxis::Pipeline};
    check(cold_spill_cell_path("/tmp", 3, pp2) != cold_spill_cell_path("/tmp", 3, WorldShape{2, 0, ParallelAxis::Pipeline}),
          "two pipeline stages must not share a spill path either");
}

// The red control for W3: a helper that drops the rank axis must be caught by the
// distinctness check.
void test_w3_control_distinctness_can_fail() {
    auto broken_path = [](std::string_view dir, std::uint32_t layer, const WorldShape&) {
        return std::string(dir) + "/ninfer_cold_L" + std::to_string(layer) + ".slot";
    };
    std::set<std::string> broken;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        broken.insert(broken_path("/tmp", 0, WorldShape{4, rank, ParallelAxis::Tensor}));
    }
    check(broken.size() == 1U,
          "the rank-blind spelling must collapse to 1 path, which is what makes the "
          "distinctness check non-vacuous");
    std::set<std::string> fixed;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        fixed.insert(cold_spill_cell_path("/tmp", 0, WorldShape{4, rank, ParallelAxis::Tensor}));
    }
    check(fixed.size() == 4U, "the rank-aware spelling must give 4 paths");
}

} // namespace

int main() {
    test_every_buildable_arch_is_handled_without_a_number_refusal();
    test_w1_control_detects_a_missing_row();
    test_record_region_budget();
    test_w2_control_the_ledger_can_fail();
    test_spill_cell_needs_a_rank_axis();
    test_w3_control_distinctness_can_fail();

    std::cout << "multidev_wiring: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
