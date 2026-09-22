// tests/test_sum_dir.cpp -- host-only self-test for src/spec/sum_dir.h
//
// Deliberately the same shape as the other header-only host tests beside it
// (tests/test_kv_tier_formats.cpp, tests/test_kv_cold_tier_budget.cpp): no CUDA, no artifact,
// no fixture. What this file is FOR:
//
//   1. section 1 pins the granularity to the tree's actual constants. src/spec/sum_dir.h
//      spells 64 itself in order to stay std-only, which is a copy -- so the copy is checked
//      here, against the two authoritative spellings, by VALUE:
//        core/paged_kv_cache.h:17        kPagedKVPageSize = 64
//        cold_host_tier.h:67-69          kColdHostPageTokens == kPagedKVPageSize (asserted)
//      Those headers are deliberately NOT included: src/core/paged_kv_cache.h:7 pulls in
//      <cuda_runtime_api.h>, and a host-only test may not require a CUDA toolchain (the same
//      rule that makes this test compile under a plain `g++`).
//   2. section 2 is the load-bearing one: IDENTITY IS CONTENT, NOT POSITION, proved rather
//      than asserted -- the same block at two offsets compares equal, and a single changed
//      token anywhere in the block does not.
//   3. sections 3-7 exercise the row, the ledger binding, the lifecycle, the wire format and
//      the idle gate, including every refusal path (a directory must refuse rather than
//      half-load).

#include "spec/sum_dir.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::spec::sum_dir::SumDir;
using ninfer::spec::sum_dir::SumDirCodec;
using ninfer::spec::sum_dir::SumDirDigest;
using ninfer::spec::sum_dir::SumDirKnobs;
using ninfer::spec::sum_dir::SumDirLoadReport;
using ninfer::spec::sum_dir::SumDirRow;
using ninfer::spec::sum_dir::SumDirState;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

template <typename A, typename B>
void check_equal(const A& got, const B& want, const char* what) {
    if (!(got == want)) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

std::vector<std::uint32_t> block_of(std::uint32_t seed, std::size_t count) {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        // An LCG, so the sequence is arbitrary but exactly reproducible.
        seed = seed * 1664525U + 1013904223U;
        tokens.push_back(seed >> 8U);
    }
    return tokens;
}

bool every_row_well_formed(const SumDir& dir) {
    for (const SumDirRow& row : dir.rows()) {
        if (!ninfer::spec::sum_dir::sum_dir_row_well_formed(row)) { return false; }
    }
    return true;
}

// ---------------------------------------------------------------------------

void section1_granularity_is_the_paged_kv_page() {
    // The directory's block must be the engine's Paged-KV page, and the engine's page is 64
    // tokens (core/paged_kv_cache.h:17 kPagedKVPageSize = 64; cold_host_tier.h:68 asserts
    // kColdHostPageTokens == kPagedKVPageSize). A change to either engine constant has to be
    // mirrored here, which is the point of duplicating the number in this test.
    check_equal(ninfer::spec::sum_dir::kSumDirBlockTokens, 64U,
                "granularity is 64 tokens (core/paged_kv_cache.h:17)");

    // The refusal policy is a compile-time property, not a runtime hope.
    check(!ninfer::spec::sum_dir::sum_dir_codec_admitted(SumDirCodec::Rejected),
          "rk4v4 has no external recall tier");
    // ADMITTED since 2026-09-18. This refusal was never sum_dir's own rule -- it is a
    // delegation to turn_recall::recall_codec_admitted (sum_dir.h:313-315) -- so correcting
    // the ledger moved this line with it. The old reason ("no single stride") named a
    // requirement no consumer of the row has: the cold IO legs are per-layer, so one
    // `file_slot` on a row addresses every layer at that layer's own extent.
    check(ninfer::spec::sum_dir::sum_dir_codec_admitted(SumDirCodec::Mixed),
          "a mixed-codec ROW is admitted: the row needs one file_slot, not one stride");
    check(!ninfer::spec::sum_dir::sum_dir_codec_admitted(SumDirCodec::Unset),
          "an unset codec may not back a row");
    check(ninfer::spec::sum_dir::sum_dir_codec_admitted(SumDirCodec::Nvfp4) &&
              ninfer::spec::sum_dir::sum_dir_codec_admitted(SumDirCodec::Int8),
          "nvfp4 and int8 are the two admitted tiers");

    // A non-page granularity is refused outright rather than silently honoured.
    bool threw = false;
    try {
        SumDir bad(1, 32);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a 32-token directory is refused (the page is the only legal block)");
}

void section2_identity_is_content_not_position() {
    const std::vector<std::uint32_t> tokens = block_of(7, 64);
    const SumDirDigest once = ninfer::spec::sum_dir::sum_dir_block_digest(tokens);
    const SumDirDigest twice = ninfer::spec::sum_dir::sum_dir_block_digest(tokens);
    check(once == twice, "the digest is a pure function of the block's content");
    check(!once.is_unset(), "a real block's digest is never the unset value");
    check(once.lo != 0 && once.hi != 0, "both lanes are pinned away from zero");

    // The same content at two positions, in two rows that differ in every bookkeeping field:
    // the IDENTITY column must still be equal, because that is the column a lookup uses.
    SumDir dir(/*sequence_tag=*/11);
    const std::size_t early = dir.append(tokens, /*token_begin=*/0, /*generation=*/11,
                                         SumDirCodec::Nvfp4, /*file_slot=*/3, /*page=*/0);
    const std::size_t late = dir.append(tokens, /*token_begin=*/64 * 40, /*generation=*/11,
                                        SumDirCodec::Nvfp4, /*file_slot=*/917, /*page=*/40);
    check_equal(dir.rows()[early].block_identity.lo, dir.rows()[late].block_identity.lo,
                "same content at two positions: identity.lo is equal");
    check_equal(dir.rows()[early].block_identity.hi, dir.rows()[late].block_identity.hi,
                "same content at two positions: identity.hi is equal");
    check(dir.rows()[early].file_slot != dir.rows()[late].file_slot,
          "the two rows really do differ in their (bookkeeping) file slot");
    check(dir.rows()[early].token_begin != dir.rows()[late].token_begin,
          "and in their (bookkeeping) token range");

    // One token changed anywhere in the block changes the identity.
    for (std::size_t index = 0; index < tokens.size(); index += 13) {
        std::vector<std::uint32_t> mutated = tokens;
        mutated[index] ^= 1U;
        check(ninfer::spec::sum_dir::sum_dir_block_digest(mutated) != once,
              "one changed token changes the block identity");
    }
    {
        std::vector<std::uint32_t> mutated = tokens;
        mutated.back() ^= 1U; // the last token: a prefix-shaped digest could miss this one
        check(ninfer::spec::sum_dir::sum_dir_block_digest(mutated) != once,
              "the last token is covered by the block identity");
    }
    {
        std::vector<std::uint32_t> shorter(tokens.begin(), tokens.begin() + 63);
        check(ninfer::spec::sum_dir::sum_dir_block_digest(shorter) != once,
              "the block's length is covered by the identity");
    }

    // A catalogue digest lives in its own domain, so a description can never be mistaken for
    // a block.
    const std::string_view description = "the retry policy discussion";
    check(ninfer::spec::sum_dir::sum_dir_text_digest(description) != once,
          "a catalogue digest is a different value from a block digest");

    // Rewriting a catalogue line does not move the identity: this is what lets the idle path
    // rewrite descriptions freely, or write them long after the rows exist.
    const SumDirDigest before = dir.rows()[early].block_identity;
    dir.set_summary(early, description);
    dir.set_summary(early, description.substr(4));
    check(dir.rows()[early].block_identity == before,
          "rewriting a catalogue line leaves the block identity alone");
}

void section3_lookup_and_the_ledger_binding() {
    SumDir dir(/*sequence_tag=*/21);
    const std::vector<std::uint32_t> alpha = block_of(101, 64);
    const std::vector<std::uint32_t> beta  = block_of(202, 64);
    const std::vector<std::uint32_t> gamma = block_of(303, 64);

    const std::size_t row_alpha = dir.append(alpha, 0, 21, SumDirCodec::Nvfp4, 3, 0);
    const std::size_t row_beta  = dir.append(beta, 64, 21, SumDirCodec::Nvfp4, 4, 1);
    const std::size_t row_gamma = dir.append(gamma, 128, 21, SumDirCodec::Int8, 5, 2);

    dir.set_summary(row_alpha, "worker queue retry backoff");
    dir.set_summary(row_beta, "retry policy and backoff ceiling");
    dir.set_summary(row_gamma, "photo of a lighthouse at dusk");
    dir.sort_rows();
    check(dir.sorted(), "sort_rows() records the ordering the lookup requires");

    const SumDirDigest alpha_id = ninfer::spec::sum_dir::sum_dir_block_digest(alpha);
    std::size_t begin = 0;
    std::size_t end   = 0;
    check(dir.find(alpha_id, begin, end), "the identity of a stored block is found");
    check_equal(end - begin, std::size_t{1}, "one row for one occurrence");

    // THE TARGETING: the term selects the two blocks that TALK ABOUT RETRY, not the block that
    // happens to sit next to the newest token. That difference is the feature.
    const std::vector<std::size_t> retry = dir.search_summaries("retry");
    check_equal(retry.size(), std::size_t{2}, "the catalogue finds the two blocks about retry");
    for (const std::size_t hit : retry) {
        check(dir.summary_of(hit).find("retry") != std::string_view::npos,
              "every hit actually carries the term");
    }

    // The ledger binding: identity AND page AND liveness all have to agree.
    check(ninfer::spec::sum_dir::sum_dir_row_bound(dir.rows()[begin], alpha_id, 0, true),
          "a live row whose ledger record matches is bound");
    check(!ninfer::spec::sum_dir::sum_dir_row_bound(dir.rows()[begin], alpha_id, 1, true),
          "a row whose page number disagrees with the ledger is not bound");
    check(!ninfer::spec::sum_dir::sum_dir_row_bound(
              dir.rows()[begin], ninfer::spec::sum_dir::sum_dir_block_digest(beta), 0, true),
          "a row whose identity disagrees with the ledger is not bound");
    check(!ninfer::spec::sum_dir::sum_dir_row_bound(dir.rows()[begin], alpha_id, 0, false),
          "a tombstoned spill region is not a hit (the ledger rule, restated in sum_dir.h)");

    // THE SLOT-IS-NOT-IDENTITY TRAP (program.h:487-518): the file slot was handed to another
    // page. A Live row may carry a stale location and must still be usable, because the
    // location is bookkeeping; what a caller may never do is key on it. The row below is the
    // Live one for `gamma` (the row indices above are pre-sort insertion order, and gamma is
    // the third insertion).
    const SumDirDigest gamma_id = ninfer::spec::sum_dir::sum_dir_block_digest(gamma);
    const SumDirRow* live_gamma = nullptr;
    for (const SumDirRow& row : dir.rows()) {
        if (row.block_identity == gamma_id && row.state == SumDirState::Live) { live_gamma = &row; }
    }
    check(live_gamma != nullptr, "the Live row for gamma is present");
    if (live_gamma != nullptr) {
        SumDirRow relocated = *live_gamma;
        relocated.file_slot = 999; // recycled / reassigned
        check(ninfer::spec::sum_dir::sum_dir_row_bound(relocated, gamma_id, 2, true),
              "a relocated file slot does not unbound a row whose identity and page agree");
        relocated.page = 7; // the page number, by contrast, IS part of the binding
        check(!ninfer::spec::sum_dir::sum_dir_row_bound(relocated, gamma_id, 2, true),
              "a reused page number does unbound the row");
        check(relocated.file_slot != live_gamma->file_slot,
              "and the two rows really do differ only in that hint");
    }
    (void)row_gamma;
}

void section4_lifecycle_is_a_tombstone_not_a_removal() {
    SumDir dir(31);
    const std::vector<std::uint32_t> tokens = block_of(404, 64);
    const std::size_t row = dir.append(tokens, 0, 31, SumDirCodec::Int8, 12, 0);
    dir.set_summary(row, "the migration checklist");
    check_equal(dir.search_summaries("migration").size(), std::size_t{1},
                "a live row is retrievable");

    check(dir.release(row), "release() reports success");
    check_equal(dir.rows()[row].state, SumDirState::Dead, "release() leaves a tombstone");
    check_equal(dir.rows()[row].file_slot, -1, "release() clears the (bookkeeping) location");
    check_equal(dir.search_summaries("migration").size(), std::size_t{0},
                "a tombstoned row is not retrievable");
    check_equal(dir.size(), std::size_t{1}, "but the row itself is kept (the identity survives)");
    check_equal(dir.summary_of(row), std::string_view{"the migration checklist"},
                "and so does its catalogue line");
    check(!ninfer::spec::sum_dir::sum_dir_row_bound(
              dir.rows()[row], ninfer::spec::sum_dir::sum_dir_block_digest(tokens), 0, true),
          "a dead row is never bound, even with a matching ledger");
    // A dead row carrying a STALE slot number is not malformed -- it is precisely the state
    // the ledger binding exists to catch -- and the binding still refuses it.
    {
        SumDirRow stale = dir.rows()[row];
        stale.file_slot = 5;
        check(ninfer::spec::sum_dir::sum_dir_row_well_formed(stale),
              "a dead row with a stale slot number is still a well-formed row");
        check(!ninfer::spec::sum_dir::sum_dir_row_bound(
                  stale, ninfer::spec::sum_dir::sum_dir_block_digest(tokens), 0, true),
              "but the binding refuses it, because it is Dead");
    }

    // A block that never spilled is Dead from the start, not a malformed Live row.
    const std::size_t never_spilled =
        dir.append(block_of(505, 64), 64, 31, SumDirCodec::Nvfp4, -1, 1);
    check_equal(dir.rows()[never_spilled].state, SumDirState::Dead,
                "a row appended with no file slot is Dead");
    check(every_row_well_formed(dir), "every row the directory appends is well-formed");
}

void section5_the_wire_format_round_trips() {
    SumDir dir(/*sequence_tag=*/4242);
    const std::vector<std::uint32_t> alpha = block_of(11, 64);
    const std::vector<std::uint32_t> beta  = block_of(22, 64);
    const std::vector<std::uint32_t> tail  = block_of(33, 17); // a short final block
    const std::size_t row_a = dir.append(alpha, 0, 4242, SumDirCodec::Nvfp4, 1, 0);
    const std::size_t row_b = dir.append(beta, 64, 4242, SumDirCodec::Nvfp4, 2, 1);
    const std::size_t row_t = dir.append(tail, 128, 4242, SumDirCodec::Nvfp4, 3, 2);
    dir.set_summary(row_a, "alpha: the ingest pipeline");
    dir.set_summary(row_b, "beta: the export pipeline");
    dir.set_summary(row_t, "tail: unexplained");
    dir.release(row_b);
    dir.sort_rows();

    const std::vector<std::uint8_t> bytes = dir.serialize();
    check(bytes.size() > ninfer::spec::sum_dir::kSumDirHeaderBytes,
          "a non-empty directory serializes past its header");

    SumDirLoadReport report;
    const SumDir back = SumDir::deserialize(bytes, report);
    check(report.header_ok, "the header seal verifies");
    check(report.payload_ok, "the payload seal verifies");
    check_equal(report.rows_read, std::uint64_t{3}, "all three rows load");
    check_equal(report.rows_rejected, std::uint64_t{0}, "no row is rejected");
    check_equal(back.sequence_tag(), std::uint64_t{4242}, "the sequence tag survives");
    check_equal(back.size(), dir.size(), "the row count survives");

    // The round trip is exact: byte-for-byte re-serialization, and field-by-field identity.
    const std::vector<std::uint8_t> again = back.serialize();
    check(again == bytes, "serialize -> deserialize -> serialize is byte-identical");
    for (std::size_t index = 0; index < dir.size(); ++index) {
        check(back.rows()[index].block_identity == dir.rows()[index].block_identity,
              "identity survives the round trip");
        check_equal(back.rows()[index].state, dir.rows()[index].state, "state survives");
        check_equal(back.rows()[index].file_slot, dir.rows()[index].file_slot,
                    "file slot survives");
        check_equal(back.rows()[index].token_begin, dir.rows()[index].token_begin,
                    "token begin survives");
        check_equal(back.rows()[index].token_end, dir.rows()[index].token_end, "token end survives");
        check_equal(back.rows()[index].page, dir.rows()[index].page, "page survives");
        check_equal(back.summary_of(index), dir.summary_of(index), "catalogue line survives");
        check(back.rows()[index].summary_digest == dir.rows()[index].summary_digest,
              "catalogue digest survives");
    }
    // The block contents survive too, so a query can still be run against the block itself.
    for (std::size_t index = 0; index < dir.size(); ++index) {
        check(back.block_content(index) == dir.block_content(index), "block content survives");
    }
    // The released row (beta, "the export pipeline") must NOT come back: a tombstone is not a hit.
    // Only alpha's line remains retrievable, and it still is after a round trip.
    check_equal(back.search_summaries("pipeline").size(), std::size_t{1},
                "lookup works after a round trip, and the tombstoned row stays excluded");
    check_equal(back.search_summaries("export").size(), std::size_t{0},
                "the tombstoned row's own line is not retrievable either");
    check_equal(back.search_summaries("ingest").size(), std::size_t{1},
                "while the live row's line still is");
}

void section6_the_loader_refuses_rather_than_half_loads() {
    SumDir dir(77);
    const std::vector<std::uint32_t> tokens = block_of(88, 64);
    const std::size_t row = dir.append(tokens, 0, 77, SumDirCodec::Nvfp4, 1, 0);
    dir.set_summary(row, "a line worth corrupting");
    dir.sort_rows();
    const std::vector<std::uint8_t> good = dir.serialize();

    const auto expect_refusal = [](const std::vector<std::uint8_t>& bytes, const char* what) {
        SumDirLoadReport report;
        const SumDir refused = SumDir::deserialize(bytes, report);
        check(refused.empty(), what);
        check(!report.error.empty(), "a refusal carries a reason");
    };

    expect_refusal(std::vector<std::uint8_t>(good.begin(), good.begin() + 12),
                   "a truncated header refuses");
    expect_refusal(std::vector<std::uint8_t>(good.begin(), good.end() - 3),
                   "a truncated payload refuses");
    {
        std::vector<std::uint8_t> bad = good;
        bad[16] ^= 0x01U; // inside the seal-covered header
        expect_refusal(bad, "a header whose seal does not verify refuses");
    }
    {
        std::vector<std::uint8_t> bad = good;
        bad[ninfer::spec::sum_dir::kSumDirHeaderBytes + 20] ^= 0x01U; // inside a row
        expect_refusal(bad, "a payload whose seal does not verify refuses");
    }
    {
        std::vector<std::uint8_t> bad = good;
        bad[4] = 99;
        expect_refusal(bad, "an unknown version refuses");
    }
    {
        std::vector<std::uint8_t> bad = good;
        bad[5] = 32;
        expect_refusal(bad, "a foreign granularity refuses");
    }
    {
        std::vector<std::uint8_t> bad = good;
        bad[6] = 1;
        expect_refusal(bad, "nonzero reserved header bits refuse");
    }
    // A refusal must not be confused with an empty directory: zero rows is a legitimate
    // artefact, and it must LOAD.
    {
        SumDir empty_dir(1);
        SumDirLoadReport report;
        const SumDir loaded = SumDir::deserialize(empty_dir.serialize(), report);
        check(report.header_ok && report.payload_ok, "a zero-row directory is valid");
        check(loaded.empty(), "and stays empty");
    }
}

void section7_the_idle_gate_is_off_by_default() {
    const SumDirKnobs defaults; // what a caller gets without asking
    check(!defaults.enabled, "the feature is OFF by default");
    check(!ninfer::spec::sum_dir::sum_dir_idle_eligible(defaults, /*drain_closed=*/true, 1 << 20),
          "a disabled feature is never idle-eligible, however long the idle");
    check(defaults.min_idle_secs > 0, "the default idle threshold is a real number of seconds");
    check(!ninfer::spec::sum_dir::sum_dir_idle_eligible(defaults, /*drain_closed=*/false, 1 << 20),
          "idle eligibility requires the drain to be closed");

    SumDirKnobs armed = defaults;
    armed.enabled = true;
    check(!ninfer::spec::sum_dir::sum_dir_idle_eligible(armed, /*drain_closed=*/false, 1 << 20),
          "an open drain is never idle, however long the window");
    check(!ninfer::spec::sum_dir::sum_dir_idle_eligible(armed, /*drain_closed=*/true,
                                                        armed.min_idle_secs - 1),
          "a too-short idle window is refused");
    check(ninfer::spec::sum_dir::sum_dir_idle_eligible(armed, /*drain_closed=*/true,
                                                       armed.min_idle_secs),
          "the threshold itself is enough, and is inclusive");
}

// ---------------------------------------------------------------------------
// section 8 -- THE UNLOAD-TEXT LAYER, and the byte-axis strategy
// ---------------------------------------------------------------------------
//
// Sections 1-7 test the DIRECTORY (identity, lookup, lifecycle, wire, idle gate). This section tests
// the layer added on top of it, the one that answers "which blocks can be unloaded and which
// cannot": the 7 verdicts and their rank contract, rule C1, the coverage audit, the byte-axis
// policy, and the one entry point that joins the directory to the sequence.
//
// The check bodies in 8.1-8.6 are the SAME checks an earlier line ran against a standalone header
// (`sum_dir_avl.h`); they are moved here so that the merged header ships with its own oracle instead
// of depending on a scratch directory. What they assert did not change -- and the message strings
// are load-bearing (they are the needles an injection harness greps for), so they are verbatim.

// The types this section names at namespace scope. (Inside the functions the same names are reached
// through `namespace A = ninfer::spec::sum_dir`, which is what keeps the section readable.)
using ninfer::spec::sum_dir::SumDirAdmissibility;
using ninfer::spec::sum_dir::SumDirBlockFacts;

// The fact-mask domain: 6 booleans, so the contract is 64 points, and all of them are driven.
// Bit order: 0 content_present, 1 inside_system_prefix, 2 overlap_frontier, 3 pinned_by_anchor,
//            4 has_control_token, 5 byte_axis_present.
SumDirBlockFacts facts_of_mask(unsigned mask) {
    SumDirBlockFacts facts;
    facts.content_present      = (mask & 1U) != 0;
    facts.inside_system_prefix = (mask & 2U) != 0;
    facts.overlap_frontier     = (mask & 4U) != 0;
    facts.pinned_by_anchor     = (mask & 8U) != 0;
    facts.has_control_token    = (mask & 16U) != 0;
    facts.byte_axis_present    = (mask & 32U) != 0;
    return facts;
}

std::vector<std::uint32_t> tokens_of(std::size_t count) {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(count);
    std::uint32_t state = 12345U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1103515245U + 12345U;
        tokens.push_back((state >> 8) & 0xFFFFU);
    }
    return tokens;
}

SumDirBlockFacts plain() { // an ordinary content block, byte axis present
    SumDirBlockFacts facts;
    facts.content_present   = true;
    facts.byte_axis_present = true;
    return facts;
}

// An INDEPENDENTLY WRITTEN predicate for "does this verdict's reason hold for these facts". It is
// deliberately a DIFFERENT formulation from the judge's: a per-VERDICT table here, versus the
// judge's per-FACT candidate list. A verdict the judge's list forgets, or a reason it invents, can
// only agree with this by accident.
bool reason_holds(SumDirAdmissibility verdict, const SumDirBlockFacts& facts) {
    switch (verdict) {
    case SumDirAdmissibility::Empty: return !facts.content_present;
    case SumDirAdmissibility::Admissible:
        return facts.content_present && facts.byte_axis_present;
    case SumDirAdmissibility::AdmissibleTextOnly:
        return facts.content_present && !facts.byte_axis_present;
    case SumDirAdmissibility::ControlToken:
        return facts.content_present && facts.has_control_token;
    case SumDirAdmissibility::PinnedByAnchor:
        return facts.content_present && facts.pinned_by_anchor;
    case SumDirAdmissibility::Resident:
        return facts.content_present && facts.overlap_frontier;
    case SumDirAdmissibility::SystemPrefix:
        return facts.content_present && facts.inside_system_prefix;
    }
    return false;
}

const SumDirAdmissibility kAllVerdicts[7] = {
    SumDirAdmissibility::Admissible,      SumDirAdmissibility::AdmissibleTextOnly,
    SumDirAdmissibility::ControlToken,    SumDirAdmissibility::PinnedByAnchor,
    SumDirAdmissibility::Resident,        SumDirAdmissibility::SystemPrefix,
    SumDirAdmissibility::Empty,
};

// A verdict vector built ONLY from the sequence: `Resident` exactly on the partial tail block,
// `AdmissibleTextOnly` on blocks the caller declares byte-less, everything else Admissible.
std::vector<SumDirAdmissibility> verdicts_from_sequence(std::uint32_t token_count,
                                                        const std::vector<std::uint32_t>& text_only) {
    const std::uint32_t blocks = ninfer::spec::sum_dir::sum_dir_block_count(token_count);
    const std::uint32_t tail   = ninfer::spec::sum_dir::sum_dir_resident_block(token_count);
    std::vector<SumDirAdmissibility> verdicts(blocks, SumDirAdmissibility::Admissible);
    for (std::uint32_t i = 0; i < blocks; ++i) {
        if (i == tail) { verdicts[i] = SumDirAdmissibility::Resident; }
    }
    for (const std::uint32_t block : text_only) {
        if (block < blocks) { verdicts[block] = SumDirAdmissibility::AdmissibleTextOnly; }
    }
    return verdicts;
}

// ---------------------------------------------------------------------------
// 8.1 -- the decision table is total, and the 64 fact combinations pin it completely
// ---------------------------------------------------------------------------
void section8a_the_decision_table_is_pinned_by_all_sixty_four_masks() {
    namespace A = ninfer::spec::sum_dir;
    using V = SumDirAdmissibility;
    check_equal(A::sum_dir_avl_rank(V::Admissible), std::uint8_t{0}, "rank admissible = 0");
    check_equal(A::sum_dir_avl_rank(V::AdmissibleTextOnly), std::uint8_t{1},
                "rank admissible-text-only = 1");
    check_equal(A::sum_dir_avl_rank(V::ControlToken), std::uint8_t{2}, "rank control-token = 2");
    check_equal(A::sum_dir_avl_rank(V::PinnedByAnchor), std::uint8_t{3}, "rank pinned = 3");
    check_equal(A::sum_dir_avl_rank(V::Resident), std::uint8_t{4}, "rank resident = 4");
    check_equal(A::sum_dir_avl_rank(V::SystemPrefix), std::uint8_t{5}, "rank system-prefix = 5");
    check_equal(A::sum_dir_avl_rank(V::Empty), std::uint8_t{6}, "rank empty = 6");
    check(A::sum_dir_avl_rank(V::Admissible) < A::sum_dir_avl_rank(V::AdmissibleTextOnly) &&
              A::sum_dir_avl_rank(V::AdmissibleTextOnly) < A::sum_dir_avl_rank(V::ControlToken) &&
              A::sum_dir_avl_rank(V::ControlToken) < A::sum_dir_avl_rank(V::PinnedByAnchor) &&
              A::sum_dir_avl_rank(V::PinnedByAnchor) < A::sum_dir_avl_rank(V::Resident) &&
              A::sum_dir_avl_rank(V::Resident) < A::sum_dir_avl_rank(V::SystemPrefix) &&
              A::sum_dir_avl_rank(V::SystemPrefix) < A::sum_dir_avl_rank(V::Empty),
          "the seven ranks are a strict total order -- no ties, no gaps");

    // The contract written once, as rank-argmax over the REASONS that hold. Deliberately NOT the
    // judge's control flow: the judge is a cascade, this is an argmax, so a reordered cascade
    // reddens here.
    unsigned disagreement = 0;
    for (unsigned mask = 0; mask < 64U; ++mask) {
        const SumDirBlockFacts facts = facts_of_mask(mask);
        const V verdict = A::sum_dir_block_admissibility(facts);
        // `have` is needed because `best` must not be pre-seeded with a rank: seeding it with
        // `Empty` (rank 6) makes every `>=` comparison false and turns the loop into a no-op.
        V best = V::Empty;
        bool have = false;
        for (const V candidate : kAllVerdicts) {
            if (!reason_holds(candidate, facts)) { continue; }
            if (!have || A::sum_dir_avl_rank(candidate) >= A::sum_dir_avl_rank(best)) {
                best = candidate;
                have = true;
            }
        }
        if (verdict != best) { ++disagreement; }
    }
    check_equal(disagreement, 0U,
                "all 64 fact combinations agree with rank-argmax (the table IS the contract)");

    SumDirBlockFacts collision;
    collision.content_present      = true;
    collision.byte_axis_present    = true;
    collision.inside_system_prefix = true;
    collision.overlap_frontier     = true;
    check_equal(A::sum_dir_block_admissibility(collision), V::SystemPrefix,
                "system prefix beats resident (5 over 4)");
    collision = plain();
    collision.overlap_frontier = true;
    collision.pinned_by_anchor = true;
    check_equal(A::sum_dir_block_admissibility(collision), V::Resident,
                "resident beats anchor (4 over 3)");
    collision = plain();
    collision.pinned_by_anchor  = true;
    collision.has_control_token = true;
    check_equal(A::sum_dir_block_admissibility(collision), V::PinnedByAnchor,
                "anchor beats control (3 over 2)");
    collision = plain();
    collision.has_control_token = true;
    collision.byte_axis_present = false;
    check_equal(A::sum_dir_block_admissibility(collision), V::ControlToken,
                "control beats text-only (2 over 1) -- THE correctness adjacency: a text-only "
                "block carrying control syntax must NOT be declared unloadable");
    collision = plain();
    collision.byte_axis_present = false;
    check_equal(A::sum_dir_block_admissibility(collision), V::AdmissibleTextOnly,
                "text-only beats admissible (1 over 0) -- the byte-axis price must stay visible");
    collision = SumDirBlockFacts{}; // nothing present, everything else set
    collision.inside_system_prefix = true;
    collision.has_control_token    = true;
    collision.byte_axis_present    = true;
    check_equal(A::sum_dir_block_admissibility(collision), V::Empty,
                "nothing to unload beats every keep reason");
}

// ---------------------------------------------------------------------------
// 8.2 -- the Resident tripwire: computable from the sequence, at most one, exactly the tail
// ---------------------------------------------------------------------------
void section8b_resident_is_decided_by_the_sequence_alone() {
    namespace A = ninfer::spec::sum_dir;
    unsigned mismatches = 0;
    for (std::uint32_t len = 0; len <= 300; ++len) {
        const std::uint32_t blocks = A::sum_dir_block_count(len);
        const std::uint32_t tail   = A::sum_dir_resident_block(len);
        const bool tail_expected   = (len != 0 && len % A::kSumDirBlockTokens != 0);

        if ((tail != A::kSumDirNoBlock) != tail_expected) { ++mismatches; }
        if (tail_expected) {
            // the tree's OWN arithmetic, not a copy of it
            if (tail != ninfer::spec::turn_recall::recall_pages_for_frontier(
                            len, A::kSumDirBlockTokens)) {
                ++mismatches;
            }
            if (tail >= blocks) { ++mismatches; } // the tail block must EXIST
        } else if (tail != A::kSumDirNoBlock) {
            ++mismatches;
        }
        const std::vector<A::SumDirBlockFactsFromSequence> facts = A::sum_dir_facts_from_sequence(len);
        if (facts.size() != blocks) { ++mismatches; }
        unsigned resident = 0;
        for (std::uint32_t i = 0; i < facts.size(); ++i) {
            if (facts[i].facts.overlap_frontier) { ++resident; }
            if (facts[i].facts.overlap_frontier != (i == tail)) { ++mismatches; }
            if (!facts[i].facts.content_present) { ++mismatches; } // Empty is unreachable
        }
        if (resident > 1) { ++mismatches; }
    }
    check_equal(mismatches, 0U,
                "for every length 0..300 the Resident block is unique, equals the tree's "
                "recall_pages_for_frontier, exists, and is the only fact the sequence can set");

    check_equal(A::sum_dir_resident_block(64), A::kSumDirNoBlock,
                "a page-aligned length has NO partial tail: nothing is forced to stay");
    check_equal(A::sum_dir_resident_block(65), 1U, "one token past 64 puts the tail at block 1");
    check_equal(A::sum_dir_committed_blocks(65), 1U,
                "and only block 0 is committed: the unload window is [0, committed_blocks)");
    check_equal(A::sum_dir_block_count(4096), 64U, "4096 tokens are 64 blocks");
}

// ---------------------------------------------------------------------------
// 8.3 -- rule C1: a maximal run, the tail is never in a text, the span is a multiple of 64
// ---------------------------------------------------------------------------
void section8c_the_builder_never_reaches_into_the_partial_tail() {
    namespace A = ninfer::spec::sum_dir;
    using V = SumDirAdmissibility;
    const std::uint32_t len = 64 * 6 + 5; // six full pages plus a 5-token tail
    const std::vector<std::uint32_t> tokens = tokens_of(len);
    std::vector<V> verdicts = verdicts_from_sequence(len, {});
    verdicts[4] = V::Resident; // a hole in the middle, so the run splits in two
    const std::vector<A::SumDirUnloadText> texts = A::sum_dir_unload_texts(verdicts, tokens);
    const A::SumDirCoverageCost cost =
        A::sum_dir_avl_audit(len, verdicts, std::vector<A::SumDirBlockFacts>{}, texts);

    check_equal(texts.size(), std::size_t{2},
                "the partial tail ends the last run: [0,4) breaks at the hole...");
    check_equal(texts[0].first_block, 0U, "the first text starts at block 0");
    check_equal(texts[0].last_block, 3U, "and stops before the hole");
    check_equal(texts[1].first_block, 5U, "the second text resumes after the hole");
    check_equal(texts[1].last_block, 5U,
                "...and [5,5] is all that is left before the partial tail");
    check_equal(cost.text_beyond_committed, 0U,
                "NO text enters the partial tail block -- the Resident block is never unloaded");
    check_equal(cost.span_not_multiple_of_block, 0U,
                "every text's span is a multiple of 64 (so one row still describes it)");
    check_equal(cost.text_beyond_sequence, 0U, "no text runs past the sequence");
    check_equal(cost.text_edge_mismatch, 0U, "every text's token edges match its block range");
    check_equal(cost.resident_blocks, 2U,
                "TWO blocks claim Resident: block 4 (the injected hole's verdict) and block 6");
    check_equal(cost.resident_misfiled, 1U,
                "and exactly one of them is a MISFILED claim: block 4 is not the tail "
                "(this is the sequence REFUTING a keep claim, with zero engine input)");
    check_equal(cost.resident_missing, 0U, "the partial tail is claimed, so nothing is missing");
    check_equal(cost.uncovered_admissible, 0U, "and no unloadable block is left uncovered");
    check_equal(cost.blocks_covered, 5U,
                "blocks 0..3 and block 5 are covered; the hole and the tail are not");
    check_equal(cost.blocks_unloadable, 5U, "five of the seven blocks are unloadable");
    check_equal(cost.uncovered_with_keep_verdict, 2U,
                "the two uncovered blocks both carry a keep verdict, so neither is a silent gap");
    check(A::sum_dir_avl_no_gap(cost), "the gap equation holds");
    check(!A::sum_dir_avl_verdicts_sound(cost),
          "but the verdicts are NOT sound -- and the gap equation was blind to it");

    // the same sequence at a page-aligned length: nothing is forced to stay
    const std::uint32_t aligned = 64 * 7;
    const std::vector<std::uint32_t> atokens = tokens_of(aligned);
    const std::vector<V> averdicts = verdicts_from_sequence(aligned, {});
    const std::vector<A::SumDirUnloadText> atexts = A::sum_dir_unload_texts(averdicts, atokens);
    const A::SumDirCoverageCost acost =
        A::sum_dir_avl_audit(aligned, averdicts, std::vector<A::SumDirBlockFacts>{}, atexts);
    check_equal(acost.resident_blocks, 0U, "at a page boundary no block is resident");
    check_equal(acost.blocks_total, 7U, "seven blocks");
    check_equal(acost.blocks_covered, 7U, "and all seven are covered by one text");
    check_equal(atexts.size(), std::size_t{1}, "one maximal run covers the whole sequence");
    check_equal(atexts[0].token_end - atexts[0].token_begin, 448U, "span 448 = 7 x 64");
    check(A::sum_dir_avl_no_gap(acost) && A::sum_dir_avl_verdicts_sound(acost),
          "gap equation AND soundness both hold at a page-aligned length");
}

// ---------------------------------------------------------------------------
// 8.4 -- the blindness, executed: a wrong KEEP verdict does not move the gap equation
// ---------------------------------------------------------------------------
void section8d_a_wrong_keep_verdict_slips_past_the_gap_equation() {
    namespace A = ninfer::spec::sum_dir;
    using V = SumDirAdmissibility;
    const std::uint32_t len = 64 * 5;
    const std::vector<std::uint32_t> tokens = tokens_of(len);

    std::vector<A::SumDirBlockFacts> facts(5, plain());
    std::vector<V> honest = A::sum_dir_admissibility_report_from(facts);
    const std::vector<A::SumDirUnloadText> honest_texts = A::sum_dir_unload_texts(honest, tokens);
    const A::SumDirCoverageCost honest_cost = A::sum_dir_avl_audit(len, honest, facts, honest_texts);
    check_equal(honest_cost.uncovered_admissible, 0U, "the honest report has no gap");
    check_equal(honest_cost.verdict_facts_mismatch, 0U, "and agrees with its own facts");
    check_equal(honest_cost.unload_texts, 1U, "five admissible blocks are ONE unload text");
    check_equal(honest_cost.covered_tokens, 320U, "covering 320 tokens");

    // CORRUPTION 1 (the one the user's question names): a block that is admissible, reported as a
    // KEEP verdict. The builder legitimately skips it and builds the two runs around it.
    std::vector<V> corrupted = honest;
    corrupted[2] = V::Resident;
    const std::vector<A::SumDirUnloadText> corrupted_texts =
        A::sum_dir_unload_texts(corrupted, tokens);
    const A::SumDirCoverageCost corrupted_cost =
        A::sum_dir_avl_audit(len, corrupted, facts, corrupted_texts);
    check_equal(corrupted_cost.uncovered_admissible, 0U,
                "BLINDNESS: a keep verdict can never be counted as uncovered_admissible, so the "
                "gap equation reads 0 while the verdict is WRONG");
    check(A::sum_dir_avl_no_gap(corrupted_cost),
          "so `uncovered_admissible == 0` still passes on a corrupted report");
    check_equal(corrupted_cost.verdict_facts_mismatch, 1U,
                "THE FIX (needs the FACTS): the report disagrees with judge(facts) on one block");
    check_equal(corrupted_cost.resident_misfiled, 1U,
                "THE FIX (needs only the SEQUENCE): block 2 is not the partial tail, so the "
                "sequence itself refutes the claim");
    check_equal(corrupted_cost.uncovered_refuted_claim, 1U,
                "and it is reported as a refuted claim, not dropped");
    check(!A::sum_dir_avl_verdicts_sound(corrupted_cost),
          "so the soundness predicate goes RED where the gap equation stayed green");

    // CORRUPTION 2: `Empty` on a block that really has content.
    std::vector<V> emptied = honest;
    emptied[1] = V::Empty;
    const A::SumDirCoverageCost emptied_cost =
        A::sum_dir_avl_audit(len, emptied, facts, A::sum_dir_unload_texts(emptied, tokens));
    check_equal(emptied_cost.empty_on_real_block, 1U,
                "`Empty` is unreachable for a block the sequence owns, so a non-zero count is a "
                "caller bug and it is caught");
    check_equal(emptied_cost.uncovered_admissible, 0U, "and the gap equation is blind here too");

    // CORRUPTION 3: `AdmissibleTextOnly`. It is NOT a keep verdict, so the block IS unloadable and
    // the text loses byte recall. This is the byte-axis price and it must be visible, not silent.
    std::vector<V> textonly = honest;
    textonly[3] = V::AdmissibleTextOnly;
    const std::vector<A::SumDirUnloadText> textonly_texts =
        A::sum_dir_unload_texts(textonly, tokens);
    const A::SumDirCoverageCost textonly_cost =
        A::sum_dir_avl_audit(len, textonly, facts, textonly_texts);
    check_equal(textonly_cost.blocks_text_only, 1U, "one block is text-only");
    check_equal(textonly_cost.unload_texts_text_only, 1U, "and the text holding it lost byte recall");
    check_equal(textonly_cost.blocks_covered, 5U, "coverage does NOT fall: the text axis still runs");
    check_equal(textonly_cost.uncovered_admissible, 0U, "and the gap equation is unaffected");
    check(!textonly_texts[0].byte_recallable, "byte_recallable propagates block -> text");

    // CORRUPTION 4 (what the gap equation IS for): the builder drops a block it should have taken.
    std::vector<A::SumDirUnloadText> holed = honest_texts;
    holed[0].last_block = 0; // ran only one block instead of five
    const A::SumDirCoverageCost holed_cost = A::sum_dir_avl_audit(len, honest, facts, holed);
    check(!A::sum_dir_avl_no_gap(holed_cost),
          "a builder gap DOES redden the gap equation -- that is what it is for");
    check_equal(holed_cost.uncovered_admissible, 4U, "four admissible blocks left uncovered");
    check(A::sum_dir_avl_verdicts_sound(holed_cost),
          "while the verdicts themselves are sound: the two numbers catch DIFFERENT faults");
}

// ---------------------------------------------------------------------------
// 8.5 -- rule C1's window is the tree's own recall page plan
// ---------------------------------------------------------------------------
void section8e_the_unload_window_is_the_trees_own_recall_page_plan() {
    namespace A = ninfer::spec::sum_dir;
    using V = SumDirAdmissibility;
    unsigned mismatches = 0;
    for (std::uint32_t len = 0; len <= 200; ++len) {
        const std::vector<V> verdicts = verdicts_from_sequence(len, {});
        const std::vector<std::uint32_t> tokens = tokens_of(len);
        const std::vector<A::SumDirUnloadText> texts = A::sum_dir_unload_texts(verdicts, tokens);

        ninfer::spec::turn_recall::RecallPagePlanRequest request;
        request.frontier    = len;
        request.page_tokens = A::kSumDirBlockTokens;
        request.page_bytes  = 4096;
        const ninfer::spec::turn_recall::RecallPagePlan plan =
            ninfer::spec::turn_recall::plan_recall_pages(
                request, [](std::uint32_t) { return true; });

        unsigned first = A::kSumDirNoBlock, last = 0;
        for (const A::SumDirUnloadText& text : texts) {
            if (text.first_block < first) { first = text.first_block; }
            if (text.last_block > last) { last = text.last_block; }
        }
        const bool any = (first != A::kSumDirNoBlock);
        if (plan.pages.size() != (any ? (last - first + 1U) : 0U)) { ++mismatches; }
        if (any && first != 0) { ++mismatches; }
        if (any && last + 1U != plan.pages.size()) { ++mismatches; }
        if (plan.dropped_clamped != 0 || plan.dropped_hole != 0) { ++mismatches; }
        for (std::size_t k = 0; k < plan.pages.size(); ++k) {
            const bool inside = any && plan.pages[k] >= first && plan.pages[k] <= last;
            if (!inside) { ++mismatches; }
        }
    }
    check_equal(mismatches, 0U,
                "for every length 0..200 the unload window is EXACTLY the tree's "
                "plan_recall_pages page set -- the text layer inherits the engine's granule");

    ninfer::spec::turn_recall::RecallPagePlanRequest request;
    request.frontier    = 64 * 3 + 9;
    request.page_tokens = A::kSumDirBlockTokens;
    const ninfer::spec::turn_recall::RecallPagePlan plan =
        ninfer::spec::turn_recall::plan_recall_pages(
            request, [](std::uint32_t) { return true; });
    check_equal(plan.pages.size(), std::size_t{3},
                "201 tokens commit three pages, not four: the partial tail is not committed");
    check_equal(plan.count(), 3U, "and the plan says so itself");
}

// ---------------------------------------------------------------------------
// 8.6 -- the byte axis is the ROW's own field, so the verdict needs no new engine hook
// ---------------------------------------------------------------------------
void section8f_the_byte_axis_is_the_rows_own_state_field() {
    namespace A = ninfer::spec::sum_dir;
    using V = SumDirAdmissibility;
    const std::uint32_t len = 64 * 3;
    const std::vector<std::uint32_t> tokens = tokens_of(len);

    // A real ledger entry: `append` DERIVES the state from `file_slot`.
    A::SumDir spilled(/*sequence_tag=*/7, A::kSumDirBlockTokens);
    const std::size_t spilled_row =
        spilled.append(tokens.data(), len, /*token_begin=*/0, /*generation=*/1,
                       A::SumDirCodec::Int8, /*file_slot=*/5, /*page=*/0);
    A::SumDir never(/*sequence_tag=*/8, A::kSumDirBlockTokens);
    const std::size_t never_row =
        never.append(tokens.data(), len, /*token_begin=*/0, /*generation=*/1,
                     A::SumDirCodec::Int8, /*file_slot=*/-1, /*page=*/0);
    check(spilled_row == 0U && never_row == 0U, "each directory took its single row");

    check_equal(static_cast<unsigned>(spilled.rows()[0].state),
                static_cast<unsigned>(A::SumDirState::Live),
                "a spilled unload-text row is Live (RecallKind::Spill)");
    check_equal(static_cast<unsigned>(never.rows()[0].state),
                static_cast<unsigned>(A::SumDirState::Dead),
                "a never-spilled row is Dead -- the row FABRICATES a Release tombstone, because "
                "turn_recall_journal.h:220-223 lists 'never written' as no cause at all");

    // The verdict reads ONLY that field, through the ONE writer.
    A::SumDirBlockFacts facts;
    facts.content_present   = true;
    facts.byte_axis_present = A::sum_dir_byte_axis_present(spilled.rows()[0]);
    check_equal(A::sum_dir_block_admissibility(facts), V::Admissible,
                "Live byte axis => fully admissible");
    facts.byte_axis_present = A::sum_dir_byte_axis_present(never.rows()[0]);
    check_equal(A::sum_dir_block_admissibility(facts), V::AdmissibleTextOnly,
                "absent byte axis => STILL UNLOADABLE, only text-only (a price, not a refusal)");
    check(A::sum_dir_is_unloadable(A::sum_dir_block_admissibility(facts)),
          "so a never-spilled block is never refused -- the dead corner is a COUPLING, not a "
          "semantic requirement");

    // ---- and the wire already carries the long span: no format change.
    A::SumDir dir(/*sequence_tag=*/99, A::kSumDirBlockTokens);
    const std::size_t long_row =
        dir.append(tokens.data(), 192, /*token_begin=*/0, /*generation=*/3, A::SumDirCodec::Int8,
                   /*file_slot=*/11, /*page=*/0);
    check(long_row == 0U, "the 192-token unload text is one row");
    const std::vector<std::uint8_t> wire = dir.serialize();
    A::SumDirLoadReport report;
    const A::SumDir back = A::SumDir::deserialize(wire, report);
    check(report.rows_read == 1U, "a 192-token unload-text row is ACCEPTED by today's loader");
    check(report.rows_rejected == 0U, "and nothing was rejected");
    check_equal(back.size(), std::size_t{1}, "and it comes back");
    check_equal(back.rows()[0].token_end - back.rows()[0].token_begin, 192U,
                "with its 192-token span intact: 3 x 64, one row, one 80-byte stride");
    check_equal(back.rows()[0].content_count, 192U, "and its 192 content tokens");
    check(back.rows()[0].block_identity == dir.rows()[0].block_identity,
          "and the SAME identity the 64-block path already computes");
    check_equal(static_cast<unsigned>(back.rows()[0].state),
                static_cast<unsigned>(A::SumDirState::Live), "still Live");

    // what the tree does NOT check: `well_formed` never looks at `content_count`, never at `page`
    A::SumDirRow liar = dir.rows()[0];
    liar.content_count = 4096;
    check(A::sum_dir_row_well_formed(liar),
          "a row claiming 4096 content tokens while carrying 192 is WELL-FORMED today "
          "(sum_dir_row_well_formed has no content_count clause)");
    A::SumDirRow bad_page = dir.rows()[0];
    bad_page.page = 12345;
    check(A::sum_dir_row_well_formed(bad_page),
          "and a row whose `page` does not equal token_begin/64 is well-formed too");
    check(liar.content_count != liar.token_end - liar.token_begin,
          "both are facts the audit must supply itself -- the loader will not");
}

// ---------------------------------------------------------------------------
// 8.7 -- THE STRATEGY: what the three byte-axis policies actually do, measured
// ---------------------------------------------------------------------------
//
// The evidence for "it is a selectable strategy, and the default changes nothing". Three things are
// pinned: the default is a VALUE (so flipping it is a red, not a silent drift); the default's fact
// IS the predicate `sum_dir_row_bound()` already applies; and the three policies produce three
// measurably different outcomes on ONE directory, so the difference is measured, not described.
void section8g_the_byte_axis_strategy_is_selectable_and_the_default_is_todays_answer() {
    namespace A = ninfer::spec::sum_dir;
    using V = SumDirAdmissibility;
    using P = A::SumDirByteAxisPolicy;

    // (i) the default, as a value and as the knobs' value
    check_equal(static_cast<unsigned>(A::kSumDirByteAxisPolicyDefault), 0U,
                "the default byte-axis policy is RefuseWhenAbsent (0) -- pinned, so a default flip "
                "is a red and not a silent behaviour change");
    check_equal(static_cast<unsigned>(A::SumDirKnobs{}.byte_axis_policy), 0U,
                "a default-constructed SumDirKnobs carries the default policy");
    check(!A::SumDirKnobs{}.byte_axis_policy_error,
          "and no policy error is reported when the flag is absent");

    // (ii) the flag parses, and an unknown value is refused rather than silently honoured
    bool ok = false;
    check_equal(static_cast<unsigned>(A::sum_dir_byte_axis_policy_parse("refuse", ok)), 0U,
                "NINFER_SUM_DIR_BYTE_AXIS=refuse parses");
    check(ok, "and is recognised");
    check_equal(static_cast<unsigned>(A::sum_dir_byte_axis_policy_parse("price", ok)), 1U,
                "NINFER_SUM_DIR_BYTE_AXIS=price parses");
    check(ok, "and is recognised");
    check_equal(static_cast<unsigned>(A::sum_dir_byte_axis_policy_parse("ignore", ok)), 2U,
                "NINFER_SUM_DIR_BYTE_AXIS=ignore parses");
    check(ok, "and is recognised");
    check_equal(static_cast<unsigned>(A::sum_dir_byte_axis_policy_parse("Refuse", ok)), 0U,
                "a misspelt value keeps the default");
    check(!ok, "AND IS REPORTED AS UNRECOGNISED -- a typo is never an invisibly different strategy");

    // (iii) the default's fact writer IS the liveness predicate the tree's own `sum_dir_row_bound`
    //       applies. This is the "the default does not change today's behaviour" claim, tied to the
    //       tree's function rather than to a restatement of it.
    {
        const std::uint32_t len3 = 64 * 3;
        const std::vector<std::uint32_t> toks = tokens_of(len3);
        A::SumDir live_dir(1, A::kSumDirBlockTokens);
        A::SumDir dead_dir(2, A::kSumDirBlockTokens);
        const std::size_t live_row =
            live_dir.append(toks.data(), len3, 0, 1, A::SumDirCodec::Int8, 5, 0);
        const std::size_t dead_row =
            dead_dir.append(toks.data(), len3, 0, 1, A::SumDirCodec::Int8, -1, 0);
        check(live_row == 0U && dead_row == 0U, "each directory took its single row");
        const A::SumDirDigest identity = A::sum_dir_block_digest(toks);
        unsigned disagreements = 0;
        unsigned live_bound = 0;
        for (const A::SumDirRow& row : live_dir.rows()) {
            const bool bound = A::sum_dir_row_bound(row, identity, 0, /*ledger_live=*/true);
            if (bound) { ++live_bound; }
            if (A::sum_dir_byte_axis_present(row) != bound) { ++disagreements; }
        }
        for (const A::SumDirRow& row : dead_dir.rows()) {
            const bool bound = A::sum_dir_row_bound(row, identity, 0, /*ledger_live=*/true);
            if (A::sum_dir_byte_axis_present(row) != bound) { ++disagreements; }
        }
        check_equal(live_bound, 1U,
                    "the live row really IS bound by sum_dir_row_bound -- so the agreement below "
                    "is not vacuous");
        check_equal(disagreements, 0U,
                    "under the DEFAULT policy the fact writer answers exactly what the tree's own "
                    "sum_dir_row_bound() answers about liveness -- the default IS today's predicate");
        check_equal(A::sum_dir_byte_axis_present(live_dir.rows()[0], P::RefuseWhenAbsent), true,
                    "a Live row has the byte axis (default policy)");
        check_equal(A::sum_dir_byte_axis_present(dead_dir.rows()[0], P::RefuseWhenAbsent), false,
                    "a Dead row does not (default policy)");
        check_equal(A::sum_dir_byte_axis_present(dead_dir.rows()[0], P::PriceWhenAbsent), false,
                    "and PriceWhenAbsent agrees about the FACT -- the policy changes the OFFERING, "
                    "never the truth");
        check_equal(A::sum_dir_byte_axis_present(dead_dir.rows()[0], P::IgnoreByteAxis), true,
                    "while IgnoreByteAxis does not consult the row at all");
        check_equal(A::sum_dir_byte_axis_present(nullptr, P::IgnoreByteAxis), true,
                    "not even when the directory has no row for the block");
        check_equal(A::sum_dir_byte_axis_present(nullptr, P::RefuseWhenAbsent), false,
                    "and 'no row' is 'no byte axis' under the default");
    }

    // (iv) the three policies on ONE directory: a 5-page sequence whose third page was never
    //      spilled. This is the measurement the strategy exists for.
    {
        const std::uint32_t len = 64 * 5;
        const std::vector<std::uint32_t> toks = tokens_of(len);
        A::SumDir dir(4242, A::kSumDirBlockTokens);
        unsigned append_mismatches = 0;
        for (std::uint32_t page = 0; page < 5; ++page) {
            // page 2 never entered a cold slot: `file_slot = -1` => the row says Dead.
            const std::size_t row_index =
                dir.append(toks.data() + page * 64, 64, page * 64, /*generation=*/4242,
                           A::SumDirCodec::Int8,
                           page == 2 ? -1 : static_cast<std::int32_t>(page), page);
            if (row_index != page) { ++append_mismatches; }
        }
        check_equal(dir.size(), std::size_t{5}, "the directory holds one row per page");
        check_equal(append_mismatches, 0U, "every append returned the row index it created");

        const A::SumDirUnloadPlan refuse =
            A::sum_dir_unload_plan(dir, len, toks, P::RefuseWhenAbsent);
        const A::SumDirUnloadPlan price = A::sum_dir_unload_plan(dir, len, toks, P::PriceWhenAbsent);
        const A::SumDirUnloadPlan ignore =
            A::sum_dir_unload_plan(dir, len, toks, P::IgnoreByteAxis);

        check_equal(static_cast<unsigned>(refuse.verdicts[2]),
                    static_cast<unsigned>(V::AdmissibleTextOnly),
                    "block 2 is AdmissibleTextOnly: its byte axis is gone, its text axis is not");
        check(refuse.verdicts == price.verdicts,
              "REFUSE and PRICE produce IDENTICAL verdicts -- they differ only in what is "
              "OFFERED, so neither one rewrites the truth");
        check(!(ignore.verdicts == price.verdicts),
              "while IGNORE produces DIFFERENT verdicts: it does not change what is offered, it "
              "changes the FACT -- which is exactly the difference between a price and a denial");

        // --- DEFAULT: refuse. The block is not offered, and the refusal is NAMED.
        check_equal(refuse.policy, P::RefuseWhenAbsent, "the default policy is what runs by default");
        check_equal(refuse.cost.blocks_text_only, 1U,
                    "REFUSE: one block's byte axis is absent (the judge sees it either way)");
        check_equal(refuse.blocks_refused_no_bytes, 1U,
                    "REFUSE: that block is refused, and COUNTED BY NAME");
        check_equal(refuse.blocks_offered, 4U, "so four of five blocks are offered");
        check_equal(refuse.texts.size(), std::size_t{2},
                    "REFUSE: the refused block is a HOLE, so rule C1 gives the two runs around it");
        check_equal(refuse.cost.blocks_covered, 4U, "coverage is four blocks, not five");
        check_equal(refuse.cost.uncovered_admissible, 1U,
                    "REFUSE: the refused block is uncovered -- so the raw gap equation reads 1");
        check(sum_dir_unload_plan_no_silent_gap(refuse),
              "...and the PLAN's equation holds, because the 1 is exactly the named refusal");
        check(!A::sum_dir_avl_no_gap(refuse.cost),
              "while the raw gap equation is NOT zero: a refusal is never silent");
        check_equal(refuse.cost.unload_texts_text_only, 0U,
                    "and no text lost byte recall: the text-only block is not in any text");

        // --- OPT-IN price: offered, and the price shows up.
        check_equal(price.cost.blocks_text_only, 1U, "PRICE sees the same single text-only block");
        check_equal(price.blocks_refused_no_bytes, 0U, "PRICE refuses nothing");
        check_equal(price.texts.size(), std::size_t{1},
                    "PRICE: one maximal run covers all five blocks -- a text-only block is still "
                    "unloadable");
        check_equal(price.cost.blocks_covered, 5U, "so coverage is five");
        check_equal(price.cost.unload_texts_text_only, 1U,
                    "and the text holding it LOSES byte recall -- the price is visible");
        check(!price.texts[0].byte_recallable, "byte_recallable propagates block -> text");
        check_equal(price.cost.uncovered_admissible, 0U, "no gap");
        check(sum_dir_unload_plan_no_silent_gap(price), "and no silent gap");
        check(A::sum_dir_avl_no_gap(price.cost), "the raw gap equation is zero too, as designed");
        check_equal(price.blocks_offered, 5U, "and every block is offered");

        // --- OPT-IN ignore: the price reads zero. This is the deleted field's failure mode, now a
        //     named strategy a reviewer can compare against instead of an accident.
        check_equal(ignore.cost.blocks_text_only, 0U,
                    "IGNORE: the price reads ZERO -- `plane_dropped`'s silent failure mode, made a "
                    "strategy you have to ask for");
        check_equal(ignore.blocks_refused_no_bytes, 0U, "IGNORE refuses nothing");
        check_equal(ignore.texts.size(), std::size_t{1}, "IGNORE: one run");
        check(ignore.texts[0].byte_recallable,
              "IGNORE: the text keeps byte recall even though the row says Dead -- the byte axis "
              "was never consulted");
        check(price.cost.unload_texts_text_only != ignore.cost.unload_texts_text_only,
              "so PRICE and IGNORE are measurably different strategies on the same directory");
        check(refuse.blocks_refused_no_bytes != price.blocks_refused_no_bytes,
              "and REFUSE differs from PRICE in the count the plan reports");

        // (v) the two builders must agree wherever the gate refuses nothing
        {
            unsigned disagreements = 0;
            for (std::uint32_t n = 0; n <= 200; ++n) {
                const std::vector<std::uint32_t> ts = tokens_of(n);
                const std::uint32_t blocks = A::sum_dir_block_count(n);
                for (std::uint32_t hole = 0; hole <= blocks; ++hole) {
                    std::vector<V> vs = verdicts_from_sequence(n, {});
                    if (hole < blocks) { vs[hole] = V::Resident; }
                    const std::vector<A::SumDirUnloadText> plain_texts =
                        A::sum_dir_unload_texts(vs, ts);
                    const std::vector<A::SumDirUnloadText> gated =
                        A::sum_dir_unload_texts_offered(vs, ts, P::RefuseWhenAbsent);
                    if (plain_texts.size() != gated.size()) { ++disagreements; continue; }
                    for (std::size_t k = 0; k < plain_texts.size(); ++k) {
                        if (plain_texts[k].first_block != gated[k].first_block ||
                            plain_texts[k].last_block != gated[k].last_block ||
                            plain_texts[k].token_begin != gated[k].token_begin ||
                            plain_texts[k].token_end != gated[k].token_end ||
                            plain_texts[k].byte_recallable != gated[k].byte_recallable) {
                            ++disagreements;
                        }
                    }
                }
            }
            check_equal(disagreements, 0U,
                        "with NO text-only block the gated builder and the plain builder return "
                        "identical texts over 0..200 x every hole: the gate is one conjunct, not a "
                        "second encoding of rule C1");
        }
    }
}

// ---------------------------------------------------------------------------
// 8.8 -- the differential corpus: inputs only, so a second implementation can be compared
// ---------------------------------------------------------------------------
void section8h_the_differential_corpus_is_non_trivial() {
    namespace A = ninfer::spec::sum_dir;
    unsigned produced = 0;
    const std::uint32_t lengths[10] = {0, 1, 63, 64, 65, 127, 128, 129, 320, 385};
    for (const std::uint32_t len : lengths) {
        const std::uint32_t blocks = A::sum_dir_block_count(len);
        const std::vector<std::uint32_t> tokens = tokens_of(len);
        for (std::uint32_t hole = 0; hole <= blocks; ++hole) {
            std::vector<SumDirAdmissibility> verdicts = verdicts_from_sequence(len, {});
            if (hole < blocks) {
                verdicts[hole] = (hole % 2 == 0) ? SumDirAdmissibility::Resident
                                                 : SumDirAdmissibility::ControlToken;
            }
            const std::vector<A::SumDirUnloadText> texts = A::sum_dir_unload_texts(verdicts, tokens);
            (void)A::sum_dir_avl_audit(len, verdicts, std::vector<A::SumDirBlockFacts>{}, texts);
            ++produced;
        }
        if (blocks > 0) {
            const std::vector<SumDirAdmissibility> verdicts = verdicts_from_sequence(len, {0});
            const std::vector<A::SumDirUnloadText> texts = A::sum_dir_unload_texts(verdicts, tokens);
            (void)A::sum_dir_avl_audit(len, verdicts, std::vector<A::SumDirBlockFacts>{}, texts);
            ++produced;
        }
    }
    check(produced >= 40U, "the differential corpus is non-trivial");
}

} // namespace

int main() {
    section1_granularity_is_the_paged_kv_page();
    section2_identity_is_content_not_position();
    section3_lookup_and_the_ledger_binding();
    section4_lifecycle_is_a_tombstone_not_a_removal();
    section5_the_wire_format_round_trips();
    section6_the_loader_refuses_rather_than_half_loads();
    section7_the_idle_gate_is_off_by_default();
    section8a_the_decision_table_is_pinned_by_all_sixty_four_masks();
    section8b_resident_is_decided_by_the_sequence_alone();
    section8c_the_builder_never_reaches_into_the_partial_tail();
    section8d_a_wrong_keep_verdict_slips_past_the_gap_equation();
    section8e_the_unload_window_is_the_trees_own_recall_page_plan();
    section8f_the_byte_axis_is_the_rows_own_state_field();
    section8g_the_byte_axis_strategy_is_selectable_and_the_default_is_todays_answer();
    section8h_the_differential_corpus_is_non_trivial();

    if (g_failures == 0) {
        std::printf("sum_dir: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "sum_dir: %d check(s) failed\n", g_failures);
    return 1;
}
