// Host-side test for src/spec/turn_recall_journal.h (P5/P6/P7 of item 6).
//
// REGISTERED: tests/CMakeLists.txt:291-292 adds this as ninfer_turn_recall_journal_test, exactly
// as :289-290 adds ninfer_lookup_fuse_test. Until 2026-09-18 this comment read "Not registered in
// the build, exactly like src/spec/lookup_fuse_test.cpp: src/spec has no CMake target at all
// today" -- false on both counts, and the second of them was exactly the kind of stale claim this
// tree keeps having to clear. It is ALSO buildable out of tree, since neither this file nor its
// header needs more than std and -I src:
//
//   g++ -std=c++20 -O2 -Wall -Wextra -I <repo>/src spec/turn_recall_journal_test.cpp -o t
//   ./t
//
// It asserts the things the design claims out loud (codec arithmetic and the rk4v4 refusal,
// the record's fixed size and crc, the tombstone rule, the torn-tail rule, the hole and
// budget cuts, and the cost anchors), and nothing that needs a GPU or the engine.

#include "spec/turn_recall_journal.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

void check_eq(std::uint64_t got, std::uint64_t want, const std::string& what) {
    if (got != want) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s: got %llu want %llu\n", what.c_str(),
                     static_cast<unsigned long long>(got),
                     static_cast<unsigned long long>(want));
    }
}

std::string temp_path(const char* name) {
    const char* dir = std::getenv("TMPDIR");
    return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

// ONE LAYER's bytes at the cold record the ENGINE'S POOL actually reserves -- 2 planes x 4
// kv_heads x the per-(page, kv_head, plane) stride. Spelled ONCE here so the seven sites below
// cannot disagree, with the authority NAMED rather than included, for the same reason
// product/kv_paging_preallocation.h:70-88 does it: this file and spec/turn_recall_journal.h are
// std-only and buildable with `-I src` alone, so it must not pull product/kv_tier_formats.h in.
// The stride is kKvColdPoolStrideBytes = 9632 B (kv_tier_formats.h:289, pinned == 9632 at :329).
// It is NOT 9536, which is the 4-BIT no-expansion record (kKvColdRansRecordAt4BitBytes, :520-521),
// and NOT 6688, the retired 2.60-bit one. Both of those stood in this file as "the real stride"
// until 2026-09-18.
constexpr std::uint64_t kKvHeadsPerLayer = 4;     // qwen3_6_27b/impl/config.h:38
constexpr std::uint64_t kPlanesPerLayer = 2;      // K and V
constexpr std::uint64_t kColdStrideBytes = 9632;  // == kv_tier_formats.h kKvColdPoolStrideBytes
constexpr std::uint64_t kLayerBytes = kPlanesPerLayer * kKvHeadsPerLayer * kColdStrideBytes;
static_assert(kLayerBytes == 77056, "2 planes x 4 kv_heads x the 9632 B cold record");

// The D1 arm's own `[textcargo] ... bytes_per_record`, KEPT AS RECORDED: 16 slot-bearing layers x
// 2 planes x 4 kv_heads x the 9,536 B record that was the nvfp4 stride when that run was captured
// on binary eedfb8aad55f8d7e (see the RECALLFIX block at the bottom of this file). It is
// deliberately NOT re-derived to today's 16 x kLayerBytes = 1,232,896: the assertion below
// reproduces a journal that was really written, and kv_bit_budget.h:413 requires a recorded cold
// value to be re-derived on purpose, never quietly "corrected".
constexpr std::uint64_t kD1PageBytes = 1220608;

void test_codec() {
    using namespace ninfer::spec::turn_recall;
    // The arithmetic is the tree's own geometry (kv_tier_formats.h:56-71 lists it; :329, :348 and
    // :521 static_assert the two records): the resident nvfp4/iso4e plane is 9216 B per head-page
    // and the int8 one 16896, while the cold records are 9632 (rANS -- the stride the pool
    // actually reserves) and 9232 (raw). 9536 is the 4-BIT no-expansion record, kept in that
    // header as the counterexample that makes the "nothing both validates and fits" clause true:
    // it is NOT a plane and NOT the stride. This file called 9536 "the real stride" until
    // 2026-09-18. 16 layers x 4 kv_heads x 256 head_dim x 2 (K,V) = 32,768 elements/token.
    check_eq(recall_codec_bytes_per_token(RecallCodec::Nvfp4), 18432, "nvfp4 B/token");
    check_eq(recall_codec_bytes_per_token(RecallCodec::Int8), 33792, "int8 B/token");
    check(kRecallNvfp4BytesPerToken < kRecallInt8BytesPerToken, "nvfp4 is the smaller tier");
    check(kRecallRk4v4BytesPerToken < kRecallNvfp4BytesPerToken, "rk4v4 is smaller still (5.6%)");
    check(recall_codec_admitted(RecallCodec::Nvfp4), "nvfp4 admitted");
    check(recall_codec_admitted(RecallCodec::Int8), "int8 admitted");
    check(!recall_codec_admitted(RecallCodec::Rejected), "rk4v4/bf16 refused");
    // The admission, as corrected on 2026-09-18. The old line here read
    // `check(!recall_codec_admitted(RecallCodec::Mixed), "a mixed-codec page is refused")`
    // and it encoded a requirement the engine does not have: both cold IO legs are
    // PER LAYER (program_impl.h:11814-11833 writes `cold_disk_files[layer]` at
    // `file_slot * cold_slots.nb[3]`; program_impl.h:12559-12579 reads that same pair) and
    // the restore's codec comes from `view.dtype`, so ONE region descriptor serves a page
    // with two strides. It is asserted in the TRUE direction now, so re-narrowing the rule
    // fails this test as well as the build (see the static_asserts in the header).
    check(recall_codec_admitted(RecallCodec::Mixed),
          "a mixed-codec page IS admitted: the cold IO legs are per-layer, so one "
          "file_slot and one layer_bytes describe a page with two strides");
    // ...and the thing that never moved: Mixed does not need a SECOND region, so the
    // descriptor budget is exactly where it was.
    check(recall_extra_regions_after_shard_axis() == 0U,
          "Mixed did not change the region budget: the record holds the PRIMARY region "
          "(@12 layer_bytes / @16 file_slot, both NAMED fields) and NO second one, because "
          "the honest pool is 4 bytes and a descriptor is 8 CONTIGUOUS ones");
    // ...and the accounting behind that 0, asserted against the ONE AUTHORITY rather than
    // against a repeated literal -- the previous form of this check hard-coded `1U`.
    check(recall_record_free_bytes_honest() ==
              recall_record_free_bytes_raw_after_shard_axis() - kRecallWordBytes,
          "the honest pool is the RAW pool minus the release reason's word: the raw form "
          "counts `reserved`@28 as free, and recall_set_release_reason() writes that whole "
          "word (live writer: program_impl.h:11351)");
    check(recall_record_free_bytes_honest() < kRecallRegionDescriptorBytes,
          "and it is smaller than one descriptor, which is WHY the count above is 0");
    check_eq(recall_codec_bytes_per_token(RecallCodec::Rejected), 0, "rejected has no payload");
    // The refusal must carry its reason, not just a false.
    check(recall_codec_refusal(RecallCodec::Rejected).find("9/448") != std::string::npos,
          "rk4v4 refusal quotes the measurement");
    check_eq(kRecallPageTokens, 64, "the tier's unit is the Paged KV page");
}

void test_record_layout() {
    using namespace ninfer::spec::turn_recall;
    check_eq(sizeof(RecallRecord), 64, "one cache line per record");
    check_eq(kRecallCrcBytes, 48, "crc covers every field before it");

    RecallRecord record{};
    record.page        = 17;
    record.layer_bytes = kLayerBytes;  // 2 planes x 4 kv_heads x the cold stride (77056)
    record.file_slot   = 3;
    record.frontier    = 18 * kRecallPageTokens;
    record.digest_lo   = 0x0123456789ABCDEFULL;
    record.digest_hi   = 0xFEDCBA9876543210ULL;
    record.codec       = static_cast<std::uint8_t>(RecallCodec::Nvfp4);
    recall_record_seal(record);
    check(recall_record_valid(record), "a sealed record validates");

    // Every covered byte is protected: corrupting the digest must invalidate the record.
    RecallRecord bad = record;
    bad.digest_lo ^= 1ULL;
    check(!recall_record_valid(bad), "a flipped digest byte fails the crc");
    bad = record;
    bad.page += 1;
    check(!recall_record_valid(bad), "a flipped page fails the crc");
    bad = record;
    bad.crc32 ^= 0xFFFFU;
    check(!recall_record_valid(bad), "a wrong crc is rejected");
    bad = record;
    bad.magic = 0;
    check(!recall_record_valid(bad), "a wrong magic is rejected");

    // Key uniqueness: same page under two different prefixes must not collide.
    check(recall_key(record.digest_lo, record.digest_hi, 17) !=
              recall_key(record.digest_lo + 1, record.digest_hi, 17),
          "the digest separates prefixes");
    check(recall_key(record.digest_lo, record.digest_hi, 17) !=
              recall_key(record.digest_lo, record.digest_hi, 18),
          "the page is part of the key");
}

void test_journal_roundtrip() {
    using namespace ninfer::spec::turn_recall;
    const std::string path = temp_path("ninfer_turn_recall_test.l0");
    std::remove(path.c_str());

    {
        TurnRecallJournal journal(JournalOpenOptions{path, true, JournalFlushPolicy::PerRecord});
        check(journal.good(), "journal opened");
        for (std::uint32_t page = 0; page < 5; ++page) {
            RecallRecord record{};
            record.page        = page;
            record.layer_bytes = kLayerBytes;
            record.file_slot   = static_cast<std::int32_t>(page);
            record.frontier    = (page + 1) * kRecallPageTokens;
            record.digest_lo   = 0x1000ULL + page;
            record.digest_hi   = 0x2000ULL;
            record.codec       = static_cast<std::uint8_t>(RecallCodec::Nvfp4);
            check(journal.append(record), "append");
        }
        // Release page 2: its file slot was handed back when it went hot again
        // (restore_cold_page:11149-11156 does exactly this today, minus the record).
        RecallRecord release{};
        release.kind      = static_cast<std::uint8_t>(RecallKind::Release);
        release.page      = 2;
        release.file_slot = -1;
        release.frontier  = 3 * kRecallPageTokens;
        release.digest_lo = 0x1002ULL;
        release.digest_hi = 0x2000ULL;
        release.codec     = static_cast<std::uint8_t>(RecallCodec::Nvfp4);
        check(journal.append(release), "append tombstone");
        check_eq(journal.records_written(), 6, "records written");
        check_eq(journal.bytes_written(), 6 * 64, "bytes written");
    }

    TurnRecallJournal reloaded(JournalOpenOptions{path, false, JournalFlushPolicy::None});
    const JournalLoadReport report = reloaded.load();
    check_eq(report.records, 6, "records replayed");
    check_eq(report.tombstones, 1, "one tombstone");
    check_eq(report.live, 4, "4 live pages after the tombstone");
    check_eq(report.torn_tail, 0, "no torn tail");
    check(reloaded.find_live(0x1000ULL, 0x2000ULL, 0) != nullptr, "page 0 is live");
    check(reloaded.find_live(0x1002ULL, 0x2000ULL, 2) == nullptr, "page 2 was tombstoned");
    check(reloaded.live_page(4), "page 4 is live");
    check(!reloaded.live_page(2), "page 2 is not live");
    check(reloaded.find_live(0x1000ULL, 0x2000ULL, 1) == nullptr, "digest+page must both match");

    const std::vector<std::uint32_t> pages = reloaded.live_pages();
    check_eq(pages.size(), 4, "live_pages size");
    check(pages.front() == 0 && pages.back() == 4, "live_pages is ascending");
    for (std::size_t i = 1; i < pages.size(); ++i) {
        check(pages[i - 1] < pages[i], "live_pages strictly ascending");
    }
    std::remove(path.c_str());
}

// A crash mid-append leaves a record that is half written; the replay must stop there
// and report it instead of trusting the bytes (the journal is a mirror).
void test_journal_torn_tail() {
    using namespace ninfer::spec::turn_recall;
    const std::string path = temp_path("ninfer_turn_recall_torn.l0");
    std::remove(path.c_str());
    {
        TurnRecallJournal journal(JournalOpenOptions{path, true, JournalFlushPolicy::None});
        RecallRecord record{};
        record.page   = 0;
        record.codec  = static_cast<std::uint8_t>(RecallCodec::Nvfp4);
        record.digest_lo = 0xAAULL;
        journal.append(record);
        journal.flush();
    }
    // Append 30 bytes of a would-be second record.
    if (std::FILE* f = std::fopen(path.c_str(), "ab")) {
        const char junk[30] = {1};
        std::fwrite(junk, 1, sizeof(junk), f);
        std::fclose(f);
    }
    TurnRecallJournal reloaded(JournalOpenOptions{path, false, JournalFlushPolicy::None});
    const JournalLoadReport report = reloaded.load();
    check_eq(report.records, 1, "one well-formed record survives");
    check_eq(report.live, 1, "the surviving record is live");
    check_eq(report.torn_tail, 1, "the truncated tail is reported, not trusted");
    std::remove(path.c_str());

    // A whole record of garbage (right length, wrong content) is also refused.
    const std::string path2 = temp_path("ninfer_turn_recall_garbage.l0");
    std::remove(path2.c_str());
    if (std::FILE* f = std::fopen(path2.c_str(), "wb")) {
        const char junk[64] = {7};
        std::fwrite(junk, 1, sizeof(junk), f);
        std::fclose(f);
    }
    TurnRecallJournal garbage(JournalOpenOptions{path2, false, JournalFlushPolicy::None});
    const JournalLoadReport garbage_report = garbage.load();
    check_eq(garbage_report.records, 0, "garbage is not a record");
    check_eq(garbage_report.live, 0, "and it is not live");
    check_eq(garbage_report.torn_tail, 1, "and it is reported");
    std::remove(path2.c_str());
}

void test_plan_hole_and_budget() {
    using namespace ninfer::spec::turn_recall;
    // 8 committed pages of 64 tokens; the wanted run is pages [2, 8).
    RecallPagePlanRequest request{};
    request.frontier    = 8 * kRecallPageTokens;
    request.page_tokens = kRecallPageTokens;
    request.page_bytes  = kLayerBytes; // one layer's bytes: this arm tests the CUT, not the unit
    request.wanted_begin_page = 2;
    request.wanted_end_page   = 8;

    // All pages live: the whole run is selected, ascending.
    const RecallPagePlan all = plan_recall_pages(request, [](std::uint32_t) { return true; });
    check_eq(all.count(), 6, "all requested pages");
    check(all.exact(), "an all-live run is exact");
    check(all.pages.front() == 2 && all.pages.back() == 7, "ascending and inclusive of the top");
    check_eq(all.bytes, 6 * kLayerBytes, "bytes = pages x page_bytes");

    // Page 5 is gone: the run is CUT at the hole, never filled.
    const RecallPagePlan holed = plan_recall_pages(
        request, [](std::uint32_t page) { return page != 5; });
    check_eq(holed.count(), 3, "pages 2..4 only");
    check_eq(holed.pages.back(), 4, "the run stops below the hole");
    // THE SPLIT. This assertion used to read only `dropped_hole == 3`, with the label
    // "pages 5..7 are counted as dropped" -- i.e. it baked in the conflation the counter
    // itself made. Page 5 is GONE; pages 6 and 7 are LIVE and cut off by the contiguity
    // rule. A total that cannot tell 1-absent from 2-cut cannot be audited, and the
    // ledger-level symptom is `hole=7` on a wanted run of 8 where only page 1 was missing.
    check_eq(holed.dropped_hole, 3, "the total cut is still reported");
    check_eq(holed.dropped_absent, 1, "exactly ONE page is absent");
    check_eq(holed.cut_above_hole(), 2, "two are LIVE and cut above the hole");
    check_eq(holed.dropped_absent + holed.cut_above_hole(), holed.dropped_hole,
             "the split accounts for the whole total, so it cannot hide a page");
    check(!holed.exact(), "a holed run is not exact");

    // A hole at the first requested page recalls nothing at all.
    const RecallPagePlan first_hole = plan_recall_pages(
        request, [](std::uint32_t page) { return page != 2; });
    check(first_hole.empty(), "a hole at the start recalls nothing");
    check_eq(first_hole.dropped_hole, 6, "and says how much it refused");
    check_eq(first_hole.dropped_absent, 1, "one page is absent");
    check_eq(first_hole.cut_above_hole(), 5, "five are live and cut above it");

    // Budget: room for two pages only.
    // ⭐ [INEXACTGATE E] THE TWO RULES ARE NOW SEPARATED, and this arm is why they had to be.
    // The ARITHMETIC (drop from the LOW end, keep the newest, stay gap-free) is unchanged and is
    // pinned below under `TruncateOldest` EXPLICITLY -- it is no longer reachable through the
    // tree's own default, because the tree's `kRecallBudgetEdgePolicy` is `RefuseNotTruncate`
    // (sum_dir_reach.h:257/:773's rule: "refusing, not truncating", which dl/vectorkey ROW 2
    // requires for a new budget). The default path is asserted separately, right after.
    RecallPagePlanRequest tight = request;
    tight.byte_budget           = 2 * kLayerBytes;
    const RecallPagePlan budget = plan_recall_pages_with_policy(
        tight, [](std::uint32_t) { return true; }, BudgetEdgePolicy::TruncateOldest);
    check_eq(budget.count(), 2, "only two pages fit");
    check(budget.pages.front() == 6 && budget.pages.back() == 7, "the newest two survive");
    check_eq(budget.dropped_budget, 4, "the oldest four are counted as dropped");
    check(!budget.exact(), "a truncated run is not exact");

    // ⭐ THE TREE'S OWN EDGE RULE, at the same request: it REFUSES the partial run and names what
    // it could not hold. This is the assertion that would have caught the shipped behaviour.
    const RecallPagePlan refused =
        plan_recall_pages(tight, [](std::uint32_t) { return true; });
    check(refused.empty(), "the tree's edge policy refuses rather than truncating");
    check(refused.defect() == RecallPagePlan::Defect::BudgetTruncated, "and names the defect");
    check_eq(refused.dropped_budget, 4, "and still says how many pages could not be held");
    check_eq(refused.dropped_budget_begin_page, 2, "and where the dropped range starts");
    check_eq(refused.dropped_budget_end_page, 6, "and where it ends");
    check_eq(refused.wanted_pages(), 6, "and what the run wanted");

    // A budget below one page selects nothing (never rounds up past a cap). Under BOTH rules:
    // there is no partial run to hold, so refusing and truncating agree here.
    RecallPagePlanRequest tiny = request;
    tiny.byte_budget           = kLayerBytes - 1;
    check(plan_recall_pages(tiny, [](std::uint32_t) { return true; }).empty(),
          "a sub-page budget recalls nothing");

    // Asking beyond the committed frontier is clamped and reported.
    RecallPagePlanRequest ahead = request;
    ahead.wanted_end_page       = 12;
    const RecallPagePlan clamped = plan_recall_pages(ahead, [](std::uint32_t) { return true; });
    check_eq(clamped.count(), 6, "clamped to the frontier");
    check_eq(clamped.dropped_clamped, 4, "and reports what was beyond it");
    check(clamped.pages.back() == 7, "the last committed page is the top");

    // The default end (0) means "up to the frontier".
    RecallPagePlanRequest open_ended = request;
    open_ended.wanted_end_page       = 0;
    open_ended.wanted_begin_page     = 0;
    check_eq(plan_recall_pages(open_ended, [](std::uint32_t) { return true; }).count(), 8,
             "0 means up to the committed frontier");
}

void test_cost_model() {
    using namespace ninfer::spec::turn_recall;
    RecallCost cost{};
    check(cost.net_positive(), "batched read beats re-prefill");
    check(cost.speedup_vs_reprefill() > 59.0 && cost.speedup_vs_reprefill() < 60.0,
          "60:1 at the conservative 6.1 us/token anchor");
    RecallCost fast{};
    fast.read_us_per_token = 2.6;
    check(fast.speedup_vs_reprefill() > 140.0 && fast.speedup_vs_reprefill() < 141.0,
          "140:1 at the 7 GB/s anchor");

    // The forbidden shape, in numbers: streaming one 1M context per step.
    RecallCost streaming{};
    streaming.tokens = 1000000;
    check(streaming.read_ms() > 2600.0, "1M tokens is > 2.6 s at 7 GB/s");
    check(!(streaming.read_ms() < streaming.decode_us_per_token / 1000.0),
          "streaming per token costs far more than one decode step (38.5 ms)");

    // The line the engine prints must carry the refusal, not hide it.
    RecallPagePlanRequest request{};
    request.frontier = 3 * kRecallPageTokens;
    request.page_bytes = kLayerBytes;
    const RecallPagePlan plan = plan_recall_pages(
        request, [](std::uint32_t page) { return page != 1; });
    const std::string line = recall_line(plan, cost, 1, 0);
    check(line.find("hole=") != std::string::npos, "the line names the hole");
    check(line.find("INEXACT(never approximated)") != std::string::npos,
          "and says what inexact means here");
    // The refusal path must name the ABSENT page and the cut ones separately: that is the
    // whole point of the split, and asserting only "hole=" would let a regression that
    // re-merges them pass.
    check(line.find("absent=1") != std::string::npos, "the line names the absent page");
    check(line.find("cut_above_hole=") != std::string::npos,
          "and how many live pages the cut cost");
}

// -------------------------------------------------------------------------------------------
// THE MEASURED LEDGER SHAPE, AS A CHECK -- RECALLFIX 2026-09-18.
//
// This is the shape D1 (COMBO1M REPORT.md section 11.1) produced at 1M on the pinned binary
// eedfb8aad55f8d7e, read straight out of the run's own journal file
// (ninfer_recall.l0: 256 Spill records over pages [0,265], and page 1 the one page in
// [0,8) with NO Spill record). The wanted run was [0,8); the plan came back as ONE page and
// the engine printed `hole=7 INEXACT(never approximated)` three times out of five rounds,
// while the sequence-end line printed `pages=5/5 refused_cost=0 digest_unavailable=0`.
//
// The check below pins the arithmetic of that shape so the two readings can never again be
// mistaken for one another: hole=7 with absent=1 is NOT "seven pages of recall are lost",
// it is "one page was never stored and six stored pages were refused by the contiguity
// rule". Those are different defects with different fixes, and the line now says which.
void test_recallfix_ledger_hole_shape() {
    using namespace ninfer::spec::turn_recall;
    RecallPagePlanRequest request{};
    request.frontier          = 8 * kRecallPageTokens;
    request.page_tokens       = kRecallPageTokens;
    request.page_bytes        = kD1PageBytes; // the engine's own bytes_per_record at D1's tier
                                              // (16 x 2 x 4 x 9,536 B), AS RECORDED -- see above
    request.wanted_begin_page = 0;
    request.wanted_end_page   = 8;

    // The ledger's own predicate: page 1 has no live Spill record, 0 and 2..7 do.
    const RecallPagePlan d1 = plan_recall_pages(
        request, [](std::uint32_t page) { return page != 1; });
    check_eq(d1.count(), 1, "the run collapses to the single page below the hole");
    check(d1.pages.size() == 1 && d1.pages.front() == 0, "and it is page 0");
    check_eq(d1.dropped_hole, 7, "the engine's own hole=7 is reproduced");
    check_eq(d1.dropped_absent, 1, "of which exactly one -- page 1 -- is absent");
    check_eq(d1.cut_above_hole(), 6, "and six (pages 2..7) are live and cut");
    check_eq(d1.dropped_absent + d1.cut_above_hole(), 7, "1 + 6 == 7");
    check_eq(d1.bytes, kD1PageBytes, "one page of bytes, not eight");
    check(!d1.exact(), "still inexact, so the refusal is still loud");

    // The line the engine prints must carry all three numbers, because each one names a
    // different fact: the cut, the page that is GONE, and the recallable pages given up.
    RecallCost cost{};
    cost.tokens = static_cast<std::uint64_t>(d1.count()) * kRecallPageTokens;
    const std::string line = recall_line(d1, cost, 1, 0);
    check(line.find("hole=7") != std::string::npos, "hole=7");
    check(line.find("absent=1") != std::string::npos, "absent=1");
    check(line.find("cut_above_hole=6") != std::string::npos, "cut_above_hole=6");
    check(line.find("INEXACT(never approximated)") != std::string::npos, "and it is loud");

    // ⭐ THE REFUSAL PATH, asserted as a refusal and not as a happy path: an all-live run
    // must print NONE of the three, so a reader who greps for `hole=` sees a clean run as
    // clean. A fix that printed `hole=0` would make every run look holed.
    const RecallPagePlan clean = plan_recall_pages(request,
                                                  [](std::uint32_t) { return true; });
    check(clean.exact(), "the all-live run is exact");
    check_eq(clean.dropped_hole, 0, "no hole");
    check_eq(clean.dropped_absent, 0, "nothing absent");
    check_eq(clean.cut_above_hole(), 0, "nothing cut");
    const std::string clean_line = recall_line(clean, cost, 8, 0);
    check(clean_line.find("hole=") == std::string::npos, "a clean run prints NO hole=");
    check(clean_line.find("absent=") == std::string::npos, "and no absent=");
    check(clean_line.find("INEXACT") == std::string::npos, "and no INEXACT");
}

} // namespace

int main() {
    test_codec();
    test_record_layout();
    test_journal_roundtrip();
    test_journal_torn_tail();
    test_plan_hole_and_budget();
    test_cost_model();
    test_recallfix_ledger_hole_shape();
    if (failures == 0) {
        std::printf("ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
