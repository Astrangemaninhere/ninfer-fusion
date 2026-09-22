// The catalogue-fidelity property of the recall index -- asserted, because nothing asserts it.
//
// WHAT THE INDEX IS MADE OF, and where the two columns diverge:
//
//   evidence  the reach selector's postings and its EXACT EXTENSION are built from
//             `SumDir::summary_of(row)`, the CATALOGUE LINE
//             (src/spec/sum_dir_reach.h PASS 1 at :439; the extension at :584-588 compares
//             `query[...] == postings.stream_tokens[...]`, and `stream_tokens` is filled from
//             `summary_of` and from nothing else).
//   restore   what a recall puts back into the context is the block's own token ids --
//             `SumDir::block_content(row)` / `sum_dir_row_tokens` / the text cargo.
//
// The engine happens to make the two agree: the producer at
// src/targets/qwen3_6/impl/runtime/program_impl.h:12591-12592 sets the line to
// `sum_dir_render_token_line(ids, count)`, the block's own ids.  But that is a property of ONE
// CALL SITE, not of the directory: `SumDir::set_summary` accepts ANY string, both retrievers
// (`SumDir::search_summaries` and `SumDirReachPostings::build`) search whatever it is given, and
// `sum_dir_row_well_formed` checks only that (summary_index != kSumDirNoSummary) ==
// (summary_count != 0).  Nothing anywhere compares the line to the content.
//
// WHY IT MATTERS, in the two directions, both measured below:
//   a line that is not the content  ->  a hit is GRANTED for a page whose content does not hold
//                                      the query: a plausible-looking WRONG ANCHOR, with the
//                                      reach line printing no refusal;
//   a line that is empty / wrong     ->  text the directory HOLDS is unreachable: a SILENT MISS
//                                      (`no-candidate`), which is the worst outcome here.
//
// Host-only, std-only, no CUDA: it compiles and runs out of tree against the two spec headers.
// Every assertion's red-ability is built in: arms B2..B5 apply a DIFFERENT catalogue line to the
// SAME content and require a DIFFERENT verdict from the SAME assertion that arm B1 satisfies.

#include "spec/sum_dir.h"
#include "spec/sum_dir_reach.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sd = ninfer::spec::sum_dir;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

constexpr std::uint32_t kBlockTokens = 64U;
// Deliberately TIGHT, and deliberately NOT the default: this arm exists to drive the
// refusal path. The shipped default is `sd::kSumDirDefaultBlocksPerPass` (sum_dir.h).
constexpr std::uint32_t kShippedCap  = 8U;

// Four synthetic blocks of 4 tokens each.  4 tokens is exactly kReachMinWindowTokens, so a
// query equal to a block's tokens is the shortest possible hit.
[[nodiscard]] std::vector<std::vector<std::uint32_t>> pages() {
    return {
        {11U, 22U, 33U, 44U},   // page 0
        {55U, 66U, 77U, 88U},   // page 1 -- shares no run of 4 with page 0
        {99U, 111U, 222U, 333U},
        {444U, 555U, 666U, 777U},
    };
}

[[nodiscard]] sd::SumDir build(const std::vector<std::vector<std::uint32_t>>& p,
                               const std::vector<std::string>& summaries) {
    sd::SumDir dir(/*sequence_tag=*/1U, kBlockTokens);
    for (std::size_t i = 0; i < p.size(); ++i) {
        const std::size_t row = dir.append(
            p[i], static_cast<std::uint32_t>(i) * kBlockTokens, /*generation=*/1U,
            sd::SumDirCodec::Int8, /*file_slot=*/static_cast<std::int32_t>(i),
            static_cast<std::uint32_t>(i));
        dir.set_summary(row, summaries[i]);
    }
    dir.sort_rows();
    return dir;
}

[[nodiscard]] std::vector<std::string> own_summaries(
    const std::vector<std::vector<std::uint32_t>>& p) {
    std::vector<std::string> out;
    for (const auto& ids : p) { out.push_back(sd::sum_dir_render_token_line(ids)); }
    return out;
}

struct Verdict {
    std::string status;
    bool found = false;
    std::uint32_t first = 0, last = 0, need = 0;
};

[[nodiscard]] Verdict ask(const sd::SumDir& dir, const std::vector<std::uint32_t>& query) {
    sd::SumDirReachPostings postings;
    postings.k_min = sd::kReachMinWindowTokens;
    postings.build(dir);
    const std::span<const std::uint32_t> none;
    const sd::SumDirReachResult r = sd::sum_dir_recall_span_reachable(
        dir, postings, sd::sum_dir_render_token_line(query),
        /*frontier=*/static_cast<std::uint32_t>(dir.size()) * kBlockTokens, kShippedCap, none);
    return Verdict{sd::sum_dir_reach_status_name(r.status), r.found, r.first_page, r.last_page,
                   r.needed_blocks};
}

// ---------------------------------------------------------------------------
// A1 -- the producer's own contract, round-tripped through the ONE renderer.
// This is the assertion that says the catalogue line carries NO information the content does
// not: if it holds for every row, the "summary" is a re-encoding of the block, not an
// abstraction OF it, and no descriptor can be read off it that the block does not already say.
// ---------------------------------------------------------------------------
void a1_renderer_is_lossless() {
    const std::vector<std::uint32_t> ids{0U, 1U, 999U, 12345U, 248076U, 1000000U, 4294967294U};
    const std::string line = sd::sum_dir_render_token_line(ids);
    std::vector<std::uint32_t> back;
    const bool parsed = sd::sum_dir_parse_token_line(line, back);
    check(parsed, "A1: the producer's line parses back");
    check(back == ids, "A1: the producer's line round-trips to the SAME ids, losslessly");
    // and it is delimited on both sides, which is what makes a substring hit an n-gram hit
    check(!line.empty() && line.front() == ' ' && line.back() == ' ',
          "A1: the line is delimited on both sides");
}

// ---------------------------------------------------------------------------
// A2..A6 -- one content, five catalogue lines, and the verdict each produces.
// ---------------------------------------------------------------------------
void a2_a6_the_five_lines() {
    const auto p = pages();
    const std::vector<std::string> own = own_summaries(p);
    const std::vector<std::uint32_t> query = p[0];   // page 0's own tokens

    // A2 -- POSITIVE CONTROL: line == the row's own tokens (what the engine's producer writes).
    {
        const sd::SumDir dir = build(p, own);
        const Verdict v = ask(dir, query);
        check(v.status == "found", "A2: with its own line, the holder is FOUND");
        check(v.first == 0U && v.last == 0U, "A2: and the run names the holder page");
    }
    // A3 -- GRANT: page 1's line is set to page 0's text.  Page 1's CONTENT is unchanged.
    {
        std::vector<std::string> s = own;
        s[1] = sd::sum_dir_render_token_line(p[0]);
        const sd::SumDir dir = build(p, s);
        const Verdict v = ask(dir, query);
        check(v.status == "found", "A3: a line that is not the content still GRANTS a hit");
        check(v.first <= 1U && 1U <= v.last, "A3: the granted run contains page 1");
        const std::vector<std::uint32_t> content1 = dir.block_content(1);
        check(content1 == p[1], "A3: page 1's CONTENT is still its own four tokens");
        check(content1 != query, "A3: and it is NOT the query -- the hit is about the LINE");
    }
    // A4 -- MISS: the holder's line is cleared.  Its content is untouched.
    {
        std::vector<std::string> s = own;
        s[0] = "";
        const sd::SumDir dir = build(p, s);
        const Verdict v = ask(dir, query);
        check(v.status == "no-candidate", "A4: an EMPTY line makes held text unreachable");
        check(dir.block_content(0) == p[0], "A4: even though the directory still HOLDS it");
    }
    // A5 -- MISS: the holder's line is set to a different page's text.
    {
        std::vector<std::string> s = own;
        s[0] = sd::sum_dir_render_token_line(p[1]);
        const sd::SumDir dir = build(p, s);
        const Verdict v = ask(dir, query);
        check(v.status == "no-candidate", "A5: a WRONG line makes held text unreachable");
    }
    // A6 -- MISS: an ABSTRACT line, which is the direction SumDirKnobs::max_summary_bytes
    // (160 B, "a catalogue line, not a paragraph") and `search_summaries`' own comment
    // ("a candidate is found by what its description says") both describe.
    {
        std::vector<std::string> s = own;
        s[0] = "a short description of the first block, well under the 160 byte knob";
        const sd::SumDir dir = build(p, s);
        const Verdict v = ask(dir, query);
        check(v.status == "no-candidate",
              "A6: an abstract line is UNSEARCHABLE by a query in the renderer's alphabet");
    }
}

// ---------------------------------------------------------------------------
// A7 -- the byte identity, reported rather than asserted: is the line smaller than the content
// it names?  A summary that is not smaller carries no abstraction.
// ---------------------------------------------------------------------------
void a7_bytes() {
    std::uint64_t line_bytes = 0, content_bytes = 0, rows = 0;
    // sweep the id range the renderer's 11 B/token reserve is sized for: a 1..10 digit id
    for (std::uint32_t width = 1; width <= 10; ++width) {
        std::uint32_t v = 1U;
        for (std::uint32_t i = 1U; i < width; ++i) { v *= 10U; }
        std::vector<std::uint32_t> ids(64U, v);
        line_bytes += sd::sum_dir_render_token_line(ids).size();
        content_bytes += 64U * 4U;
        ++rows;
    }
    std::printf("# A7 catalogue line vs content, 64-token blocks, ids of 1..10 digits:\n"
                "#     line=%llu B  content=%llu B  ratio=%.3f x  (the engine's reserve is 11 "
                "B/token = 704 B)\n",
                static_cast<unsigned long long>(line_bytes),
                static_cast<unsigned long long>(content_bytes),
                static_cast<double>(line_bytes) / static_cast<double>(content_bytes));
    check(line_bytes > 0U && rows == 10U, "A7: the byte sweep ran");
}

} // namespace

int main() {
    std::printf("# test_recall_catalogue_fidelity  kReachMinWindowTokens=%u  shipped cap=%u\n",
                sd::kReachMinWindowTokens, kShippedCap);
    a1_renderer_is_lossless();
    a2_a6_the_five_lines();
    a7_bytes();
    std::printf("# CHECKS=%d FAILURES=%d %s\n", g_checks, g_failures,
                g_failures == 0 ? "ALL PASSED" : "*** FAILED ***");
    return g_failures;
}
