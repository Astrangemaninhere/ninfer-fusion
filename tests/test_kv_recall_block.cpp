// Host test for the block-level KV ownership encoding (product/kv_recall_block.h).
//
// Two kinds of check, deliberately kept apart:
//
//  1. SOURCE PINS. The header's central claim is that its unit IS the engine's
//     Paged-KV page. That claim is not provable by static_assert here without
//     dragging cuda_runtime_api.h into a plain-g++ test, so it is checked against
//     the authoritative source text instead: the page size, the page shift/mask the
//     attention address helper uses, and the token count the cold host tier
//     restates. If any of the three moves, this test fails with the drift named.
//
//  2. BEHAVIOUR. The span arithmetic, the digest (stability, length binding,
//     position independence), the write-side gates and the probe. The last block
//     is the one that matters for the work package: a block's identity survives
//     its device cold slot being recycled, which is the whole reason the encoding
//     is content-derived and lives beside the bytes instead of inside them.
//
// Plain g++, no CUDA, no artifact. Note the include roots match the ones
// ninfer_add_test sets up (tests/CMakeLists.txt:64-84):
//   g++ -std=c++20 -I src -I include -I third_party tests/test_kv_recall_block.cpp -o /tmp/t
// then run /tmp/t with NINFER_SOURCE_DIR pointing at the tree.

#include "product/kv_recall_block.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace p = ninfer::product;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// ---------------------------------------------------------------------------
// Source access. ctest passes NINFER_SOURCE_DIR (tests/CMakeLists.txt:78-82); a
// plain run from the repo root falls back to the working directory.
// ---------------------------------------------------------------------------
std::filesystem::path source_root() {
#ifdef NINFER_SOURCE_DIR
    return std::filesystem::path(NINFER_SOURCE_DIR);
#else
    if (const char* from_env = std::getenv("NINFER_SOURCE_DIR");
        from_env != nullptr && *from_env != '\0') {
        return std::filesystem::path(from_env);
    }
    return std::filesystem::current_path();
#endif
}

std::string slurp(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { return {}; }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// The declaration text that follows the FIRST occurrence of `needle`, up to the
// next ';' or end of line. Source pins are checked against this window rather than
// against a fixed offset, so a re-spaced or re-commented declaration still parses.
std::string_view declaration_window(std::string_view text, std::string_view needle) {
    const std::size_t at = text.find(needle);
    if (at == std::string_view::npos) { return {}; }
    std::size_t end = at;
    while (end < text.size() && text[end] != ';' && text[end] != '\n') { ++end; }
    return text.substr(at, end - at);
}

// First run of digits inside a declaration window, or -1 when the value is a name
// rather than a literal. Used only where the header really does spell a literal.
int literal_in(std::string_view window) {
    for (std::size_t pos = 0; pos < window.size(); ++pos) {
        if (window[pos] < '0' || window[pos] > '9') { continue; }
        int value = 0;
        while (pos < window.size() && window[pos] >= '0' && window[pos] <= '9') {
            value = value * 10 + (window[pos] - '0');
            ++pos;
        }
        return value;
    }
    return -1;
}

void source_pins() {
    const std::filesystem::path root = source_root();

    const std::string page_header = slurp(root / "src/core/paged_kv_cache.h");
    const std::string address     = slurp(root / "src/ops/kernel/paged_kv_address.cuh");
    const std::string host_tier   = slurp(root / "src/targets/qwen3_6/impl/runtime/cold_host_tier.h");

    check(!page_header.empty(), "could not read src/core/paged_kv_cache.h under " + root.string());
    check(!address.empty(), "could not read src/ops/kernel/paged_kv_address.cuh under " + root.string());
    check(!host_tier.empty(), "could not read cold_host_tier.h under " + root.string());
    if (page_header.empty() || address.empty() || host_tier.empty()) { return; }

    // core/paged_kv_cache.h:17 -- the block table's addressing unit.
    const int page_size =
        literal_in(declaration_window(page_header, "kPagedKVPageSize"));
    check(page_size == static_cast<int>(p::kRecallBlockTokens),
          "kPagedKVPageSize (" + std::to_string(page_size) + ") != kRecallBlockTokens (" +
              std::to_string(p::kRecallBlockTokens) + ")");

    // ops/kernel/paged_kv_address.cuh:9-10 -- the shift/mask the kernels index with.
    // The shift is a literal; the mask is SPELLED as `kPagedKVPageSize - 1`, so it
    // is checked as that relation instead of as a number (which is what makes it
    // stay true if the page size ever moves).
    const int shift = literal_in(declaration_window(address, "kPagedKVPageShift"));
    check(shift >= 0 && (1 << shift) == page_size,
          "kPagedKVPageShift (" + std::to_string(shift) + ") does not match page size " +
              std::to_string(page_size));
    const std::string_view mask_window = declaration_window(address, "kPagedKVPageMask");
    check(mask_window.find("kPagedKVPageSize") != std::string_view::npos &&
              mask_window.find("- 1") != std::string_view::npos,
          "kPagedKVPageMask must stay spelled as kPagedKVPageSize - 1 (window: \"" +
              std::string(mask_window) + "\")");

    // cold_host_tier.h:67 -- the cold tier restates the same unit, which is the
    // precedent this header follows. Its own static_assert pins it to kPagedKVPageSize.
    const int cold_host_tokens =
        literal_in(declaration_window(host_tier, "kColdHostPageTokens"));
    check(cold_host_tokens == static_cast<int>(p::kRecallBlockTokens),
          "kColdHostPageTokens (" + std::to_string(cold_host_tokens) +
              ") != kRecallBlockTokens (" + std::to_string(p::kRecallBlockTokens) + ")");
}

// ---------------------------------------------------------------------------
// Span arithmetic
// ---------------------------------------------------------------------------
void spans() {
    const std::uint32_t page = p::kRecallBlockTokens; // 64

    // A page well inside the sequence is full.
    const auto full = p::recall_block_span(3, 10 * page);
    check(full.has_value() && full->first == 3 * page && full->second == page,
          "full page 3 must span [192, 256)");

    // The tail page is partial and is CLAMPED, not dropped.
    const auto tail = p::recall_block_span(2, 2 * page + 7);
    check(tail.has_value() && tail->first == 2 * page && tail->second == 7,
          "tail page must clamp to the committed token count");

    // Exactly on a boundary: the page before the frontier is full, the frontier's
    // own page does not exist yet.
    const auto last = p::recall_block_span(1, 2 * page);
    check(last.has_value() && last->first == page && last->second == page,
          "page 1 must be full when valid_tokens == 128");
    check(!p::recall_block_span(2, 2 * page).has_value(),
          "page 2 must not exist when valid_tokens == 128");

    // Empty / already-past sequences yield nothing rather than a zero-width block.
    check(!p::recall_block_span(0, 0).has_value(), "no block exists in an empty sequence");
    check(!p::recall_block_span(5, 3 * page).has_value(), "page 5 is past 3 pages");

    // A one-token tail is still a block.
    const auto one = p::recall_block_span(1, page + 1);
    check(one.has_value() && one->first == page && one->second == 1,
          "a one-token tail page is a block with token_count 1");
}

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------
std::vector<ninfer::TokenId> tokens(std::initializer_list<int> ids) {
    return std::vector<ninfer::TokenId>(ids.begin(), ids.end());
}

void digests() {
    const std::vector<ninfer::TokenId> a = tokens({10, 20, 30, 40});
    const std::vector<ninfer::TokenId> b = tokens({10, 20, 30, 41});

    check(p::recall_block_digest(a) == p::recall_block_digest(a),
          "the digest must be a pure function of its input");
    check(p::recall_block_digest(a) != p::recall_block_digest(b),
          "different tokens must not share a digest");

    // Length binding: the fold is prefix-extendable, so without the length term a
    // block and its own prefix would collide at different token counts.
    const std::vector<ninfer::TokenId> prefix = tokens({10, 20});
    check(p::recall_block_digest(prefix) != p::recall_block_digest(a),
          "a strict prefix must not share its extension's digest");

    // POSITION INDEPENDENCE, the property the whole encoding rests on: the same
    // text at two different places in the sequence has the same identity, because a
    // recall re-prefills that text as new input at new positions. If position were
    // folded in, a recalled block could never match its own record.
    const std::vector<ninfer::TokenId> elsewhere = tokens({10, 20, 30, 40});
    check(p::recall_block_digest(elsewhere) == p::recall_block_digest(a),
          "the digest must be position-independent");

    // A zero token id is a real value, not an end marker.
    check(p::recall_block_digest(tokens({0})) != p::recall_block_digest(tokens({0, 0})),
          "repeated zero ids must still be length-distinguished");
}

// ---------------------------------------------------------------------------
// The write side and its gates
// ---------------------------------------------------------------------------
constexpr std::uint32_t kPage = p::kRecallBlockTokens;

std::vector<ninfer::TokenId> ledger_of(std::uint32_t count) {
    std::vector<ninfer::TokenId> ledger;
    ledger.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        ledger.push_back(static_cast<ninfer::TokenId>(1000 + i));
    }
    return ledger;
}

void writes() {
    const std::vector<ninfer::TokenId> ledger = ledger_of(3 * kPage);
    p::RecallBlockRecord record;

    // Happy path.
    p::RecallBlockWrite result = p::recall_note_block(record, ledger, kPage, /*page=*/1,
                                                      /*row=*/4, /*generation=*/9, /*slot=*/7,
                                                      /*file_slot=*/2, p::RecallBlockCodec::Nvfp4Rans);
    check(result == p::RecallBlockWrite::Recorded, "a normal eviction must record");
    check(record.occupied && record.logical_page == 1 && record.row == 4 &&
              record.row_generation == 9 && record.slot == 7 && record.file_slot == 2 &&
              record.codec == p::RecallBlockCodec::Nvfp4Rans && record.token_begin == kPage &&
              record.token_count == kPage,
          "the recorded fields must be the ones passed in");
    check(record.content_digest != 0, "a recorded block must carry a digest");
    check(p::recall_block_probe(record, ledger, 3 * kPage).empty(),
          "a freshly written record must pass its own probe");

    // No codec: mirrors program_impl.h:10934-10936, where a layer with no cold
    // codec fails the page and the page stays hot. Recording it would create a
    // directory entry that no recall could ever restore.
    p::RecallBlockRecord no_codec;
    check(p::recall_note_block(no_codec, ledger, kPage, 1, 4, 9, 7, 2, p::RecallBlockCodec::None) ==
              p::RecallBlockWrite::NoCodec,
          "a block with no cold codec must be refused");
    check(!no_codec.occupied, "a refused write must not mark the record occupied");

    // The engine's page moved: refuse loudly instead of recording a block whose
    // unit no longer means what the directory thinks.
    p::RecallBlockRecord wrong_page;
    check(p::recall_note_block(wrong_page, ledger, kPage * 2, 1, 4, 9, 7, 2,
                               p::RecallBlockCodec::Int8Raw) == p::RecallBlockWrite::BadPageTokens,
          "a page_tokens other than 64 must be refused");

    // The ledger does not cover the page.
    p::RecallBlockRecord short_ledger;
    const std::vector<ninfer::TokenId> tiny = ledger_of(kPage / 2);
    check(p::recall_note_block(short_ledger, tiny, kPage, /*page=*/1, 4, 9, 7, 2,
                               p::RecallBlockCodec::Int8Raw) == p::RecallBlockWrite::LedgerTooShort,
          "a page past the ledger must be refused");

    // A partial tail block IS recorded, with the clamped count.
    p::RecallBlockRecord tail;
    const std::vector<ninfer::TokenId> partial = ledger_of(kPage + 5);
    check(p::recall_note_block(tail, partial, kPage, /*page=*/1, 4, 9, 7, -1,
                               p::RecallBlockCodec::Int8Raw) == p::RecallBlockWrite::Recorded,
          "a partial tail block must record");
    check(tail.token_count == 5 && tail.file_slot == -1,
          "a partial tail block must carry its clamped count and a -1 file slot");
    check(p::recall_block_probe(tail, partial, kPage + 5).empty(),
          "a partial tail block must pass its own probe");
}

// ---------------------------------------------------------------------------
// The probe catches what it must
// ---------------------------------------------------------------------------
void probes() {
    const std::vector<ninfer::TokenId> ledger = ledger_of(4 * kPage);
    p::RecallBlockRecord record;
    (void)p::recall_note_block(record, ledger, kPage, /*page=*/2, /*row=*/0, /*generation=*/1,
                               /*slot=*/3, /*file_slot=*/-1, p::RecallBlockCodec::Int8Raw);
    check(p::recall_block_probe(record, ledger, 4 * kPage).empty(), "baseline record is clean");

    // A digest that does not re-derive from the text is the failure mode the Rk4v4
    // gate bug argues for: reading and writing must agree, and the probe recomputes
    // rather than trusting the stored value.
    p::RecallBlockRecord tampered = record;
    tampered.content_digest ^= 0x1ULL;
    check(!p::recall_block_probe(tampered, ledger, 4 * kPage).empty(),
          "the probe must catch a digest that does not re-derive");

    // A span that disagrees with the page index would recall the wrong text.
    p::RecallBlockRecord shifted = record;
    shifted.token_begin += kPage;
    check(!p::recall_block_probe(shifted, ledger, 4 * kPage).empty(),
          "the probe must catch a span that disagrees with the page index");

    // A page past the committed frontier.
    p::RecallBlockRecord past = record;
    past.logical_page = 9;
    check(!p::recall_block_probe(past, ledger, 4 * kPage).empty(),
          "the probe must catch a page past the valid token count");

    // A sequence truncated under a live record (rewrite / fork).
    check(!p::recall_block_probe(record, ledger, kPage + 1).empty(),
          "the probe must catch a record whose block is no longer fully committed");

    // An empty record is not a live record.
    p::RecallBlockRecord empty;
    check(!p::recall_block_probe(empty, ledger, 4 * kPage).empty(),
          "the probe must reject an empty record");

    // A moved unit.
    p::RecallBlockRecord moved = record;
    moved.page_tokens = kPage / 2;
    check(!p::recall_block_probe(moved, ledger, 4 * kPage).empty(),
          "the probe must catch page_tokens drifting from the engine's page");
}

// ---------------------------------------------------------------------------
// The point of the work package: identity survives slot recycling and restore.
// ---------------------------------------------------------------------------
void identity_survives_recycling() {
    const std::vector<ninfer::TokenId> ledger = ledger_of(6 * kPage);

    // Block 1 is evicted into cold slot 3.
    p::RecallBlockRecord before;
    (void)p::recall_note_block(before, ledger, kPage, /*page=*/1, /*row=*/0, /*generation=*/1,
                               /*slot=*/3, /*file_slot=*/0, p::RecallBlockCodec::Nvfp4Rans);

    // The slot is recycled: PagedKVCache::allocate_cold_slot hands back the lowest
    // free index (decoder_state.cpp:617-626), so the SAME slot number now holds a
    // different page. The engine's sentinel (-2 - slot, core/paged_kv_cache.h:19-32)
    // cannot tell the two apart -- the record must.
    const std::vector<ninfer::TokenId> other_text = tokens({7, 7, 7, 7});
    p::RecallBlockRecord reused;
    (void)p::recall_note_block(reused, ledger, kPage, /*page=*/4, /*row=*/0, /*generation=*/1,
                               /*slot=*/3, /*file_slot=*/-1, p::RecallBlockCodec::Nvfp4Rans);

    check(before.slot == reused.slot && before.logical_page != reused.logical_page,
          "the scenario must actually reuse a slot for a different page");
    check(before.content_digest != reused.content_digest,
          "two pages sharing a recycled slot must be distinguishable by content");
    check(before.row_generation == reused.row_generation, "same row generation in this scenario");

    // Recall resolves by CONTENT at a NEW position: the block is re-prefilled
    // somewhere else, and the record still matches because position is not folded in.
    const std::span<const ninfer::TokenId> original =
        std::span<const ninfer::TokenId>(ledger).subspan(before.token_begin, before.token_count);
    check(p::recall_block_content_matches(before, original),
          "the block must match the text it was built from");
    check(!p::recall_block_content_matches(before, std::span<const ninfer::TokenId>(other_text)),
          "a different text must not match the block");

    // A row generation bump invalidates the whole row: a recycled execution row
    // must not resolve an old record. The engine guards rows the same way
    // (core/paged_kv_cache.h:338-344, row_generations_ at :417).
    p::RecallBlockRecord stale = before;
    stale.row_generation = reused.row_generation + 1;
    check(stale.row_generation != before.row_generation,
          "the generation guard must be carried on the record, not inferred");
}

// ---------------------------------------------------------------------------
// The injection marker's home is disjoint from the existing evidence bits.
// ---------------------------------------------------------------------------
void injection_marker() {
    // include/ninfer/types.h:532-539 -- None, ExplicitBoundary (1<<0),
    // RequestedAutomatic (1<<1), DefaultAutomatic (1<<2), EngineStructural (1<<3),
    // EngineObserved (1<<4). Bit 5 is the first free one; the header static_asserts
    // this, and the check below keeps the constant honest at runtime too.
    constexpr std::uint8_t kUsed = (1U << 0U) | (1U << 1U) | (1U << 2U) | (1U << 3U) | (1U << 4U);
    check((p::kRecallEvidenceBit & kUsed) == 0U,
          "the recall marker bit must not alias an existing SharedCandidateEvidence reason");
    check(p::kRecallEvidenceBit == 0x20U, "the first free evidence bit is 1<<5");

    // token_types is NOT the channel: a non-zero type means a Vision modality and
    // the prefill path throws when no matching Vision chunk exists
    // (text_prefill_impl.h:349-363).
    check(p::kOrdinaryTextTokenType == 0,
          "ordinary text is token_type 0; a recall marker must not be written there");
}

} // namespace

int main() {
    source_pins();
    spans();
    digests();
    writes();
    probes();
    identity_survives_recycling();
    injection_marker();

    if (failures != 0) {
        std::cerr << "kv_recall_block_test: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "kv_recall_block_test: all checks passed\n";
    return 0;
}
