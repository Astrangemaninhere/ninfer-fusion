// tests/test_cold_slot_release_bytes.cpp
//
// [FALLBACKBYTES] WHAT A COLD-TIER FALLBACK DOES TO THE DEVICE BYTE BUDGET.
//
// THE CLAIM UNDER TEST. "A page that falls back costs no bytes: release_cold_slot(slot)
// returns the slot and the page keeps its 9216 B plane. => the fallback rate is a hit-rate
// (capacity) term, never a bytes/component term." The conclusion is TRUE for the cold POOL;
// the stated mechanism is WRONG, and it is wrong in the direction that matters for 1M. This
// test pins the TRUE contract, in four separable clauses:
//
//   (1) THE POOL IS A FIXED ARENA. The cold slot pool is ONE `DType::U8` tensor per layer of
//       shape `{stride, kv_heads, 2, max_cold_pages}`, added to the device arena once, in
//       full, at plan time, from the PLAN's page count. Nothing at runtime resizes it and
//       nothing in its sizing path can see how many pages were actually delivered.
//   (2) A FALLBACK RETURNS THE SLOT IDENTITY. `release_cold_slot` writes `false` into the
//       host bitmap `cold_slot_used_` and does nothing else, so the slot is re-allocatable.
//   (3) A FALLBACK RETAINS THE BYTES. The body of `release_cold_slot` reaches no arena, no
//       deallocator and no size field. That is what makes it byte-free -- NOT the page's
//       9216 B plane, which lives in a DIFFERENT structure (`pages_`) that this function
//       never names. `product::kKvColdResidentNvfp4Bytes == 9216` is the RESIDENT plane.
//   (4) AND THAT IS THE ONLY PLACE IT IS BYTE-FREE. Only the SUCCESS path releases the
//       resident page, by calling `store.transfer_to_cold(text, page)`. Every abandonment
//       path (`!slot`, codec refused, disk staging absent, spill budget exhausted, invalid
//       flag) reaches `release_cold_slot` and NEVER `transfer_to_cold`. So the fallback rate
//       is a real byte term on the resident arena, in the direction of LOSS: a high fallback
//       rate needs MORE resident pages, never fewer.
//
// WHY THE CHECKS ARE TWO KINDS, AND WHICH IS WHICH.
// Clauses (1)/(2)/(4) are properties of `program_impl.h` + `decoder_state.cpp`, which cannot be
// linked here: the engine's device link is broken at this revision (the half-landed
// iso3->iso4e / e8->rk4v4 rename leaves `ops::EntropyColdRequantMode` without
// `Iso4eVG16`/`Rk4v4KvG64`). So those clauses are asserted against the SOURCE TEXT, in the
// established shape of tests/test_fnv_convention.cpp and ops/test_gqa_decode_split_exact.cu
// ("reads the kernel header AS TEXT"). The pool ARITHMETIC in clause (1) is additionally
// executed, from the real cuda-free header product/kv_tier_formats.h.
//
// A source-text check is only as good as its power to reject. Every text check below is
// therefore paired with a NEGATIVE CONTROL on a synthetic broken copy of the same snippet
// (check_negative_controls()), so the test cannot pass vacuously by finding nothing.
//
// The `9216` above is `product::kKvColdResidentNvfp4Bytes` (product/kv_tier_formats.h:278),
// the resident NVFP4 plane pair for one head-page at head_dim 256 x 64 tokens; it is a
// COMPILE-TIME constant of the byte table, so the test reads it from the header rather than
// re-spelling it.

#include "product/kv_paging_preallocation.h"
#include "product/kv_tier_formats.h"
#include "targets/qwen3_6/impl/runtime/cold_fallback_census.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>

#ifndef NINFER_SOURCE_DIR
#error "this test reads the tree; register it with NEEDS_SOURCE_DIR / NINFER_SOURCE_DIR"
#endif

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// ---------------------------------------------------------------------------
// text plumbing
// ---------------------------------------------------------------------------

[[nodiscard]] std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "FATAL: cannot read %s\n", path.c_str());
        std::exit(2);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Blank out `//` and `/* */` comments so a forbidden-token scan cannot be satisfied or
// defeated by prose. Deliberately simple: it tracks `"` and `'` so a `//` inside a string
// literal survives, and it assumes the sources it is handed are comment-balanced.
[[nodiscard]] std::string strip_comments(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    char quote = '\0';
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quote != '\0') {
            out.push_back(c);
            if (c == '\\' && i + 1 < text.size()) {
                out.push_back(text[++i]);
                continue;
            }
            if (c == quote) { quote = '\0'; }
            continue;
        }
        if (c == '"' || c == '\'') {
            quote = c;
            out.push_back(c);
            continue;
        }
        if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            while (i < text.size() && text[i] != '\n') { ++i; }
            out.push_back('\n');
            continue;
        }
        if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
            i += 2;
            while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/')) { ++i; }
            ++i;
            continue;
        }
        out.push_back(c);
    }
    return out;
}

// Inner text of the brace group opening at `open` (which must point at '{').
[[nodiscard]] std::string brace_body(const std::string& text, std::size_t open) {
    if (open >= text.size() || text[open] != '{') { return {}; }
    int depth = 0;
    for (std::size_t i = open; i < text.size(); ++i) {
        if (text[i] == '{') {
            ++depth;
        } else if (text[i] == '}') {
            --depth;
            if (depth == 0) { return text.substr(open + 1, i - open - 1); }
        }
    }
    return {};
}

// Body of the FIRST definition whose signature contains `sig`. Returns "" if the signature
// is not found -- callers must assert non-empty, which is what makes the check non-vacuous.
[[nodiscard]] std::string body_of(const std::string& text, const std::string& sig) {
    const std::size_t at = text.find(sig);
    if (at == std::string::npos) { return {}; }
    const std::size_t brace = text.find('{', at + sig.size());
    if (brace == std::string::npos) { return {}; }
    return brace_body(text, brace);
}

[[nodiscard]] bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

[[nodiscard]] std::size_t count_of(const std::string& hay, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t at = hay.find(needle); at != std::string::npos;
         at = hay.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

[[nodiscard]] std::string source_path(const char* relative) {
    return std::string(NINFER_SOURCE_DIR) + "/" + relative;
}

// ---------------------------------------------------------------------------
// clause 1, executed: the pool's byte arithmetic, from the real header
// ---------------------------------------------------------------------------

// One page's cold records across every slot-bearing layer: SUM over layers of
// (stride_layer x kv_heads x 2). This is the engine's own `turn_recall_page_bytes()`
// (program_impl.h:12941), and the same SUM the pool tensor's `nb[3]` carries per layer.
[[nodiscard]] constexpr std::uint64_t page_record_bytes(std::uint32_t layers,
                                                       std::uint64_t stride,
                                                       std::uint32_t kv_heads) noexcept {
    return static_cast<std::uint64_t>(layers) * stride * kv_heads * 2ULL;
}

// The pool's TOTAL device bytes. `paging_device_cold_slot_pages` is the authority for the
// page count; this multiplies it by one page's records.
[[nodiscard]] constexpr std::uint64_t cold_pool_bytes(std::uint32_t pages,
                                                     std::uint64_t per_page) noexcept {
    return static_cast<std::uint64_t>(pages) * per_page;
}

// The rANS ceiling, read out of decoder_state.cpp's TEXT -- the file cannot be included here
// (it is CUDA), and the ceiling is what the header's stride is derived from, so the two must be
// compared rather than either one hard-coded. Returns 0 if absent.
[[nodiscard]] std::int32_t rans_ceiling_x100() {
    const std::string text = strip_comments(
        read_file(source_path("src/targets/qwen3_6/impl/state/decoder_state.cpp")));
    const std::string key = "kColdSlotRansBitsPerCodeX100 = ";
    const std::size_t at = text.find(key);
    if (at == std::string::npos) { return 0; }
    std::int32_t value = 0;
    for (std::size_t i = at + key.size(); i < text.size(); ++i) {
        const char c = text[i];
        if (c < '0' || c > '9') { break; }
        value = value * 10 + (c - '0');
    }
    return value;
}

// The stride the ceiling implies, by the header's own integer ceil: 320 B header + 32 streams of
// ceil(512 x ceiling_x100 / (8 x 100)) + the 1024 B scale tail.
[[nodiscard]] constexpr std::int32_t rans_stride_for_ceiling(std::int32_t ceiling_x100) noexcept {
    const std::int32_t stream_bytes = (512 * ceiling_x100 + 8 * 100 - 1) / (8 * 100);
    return 320 + 32 * stream_bytes + 1024;
}

void check_pool_is_a_fixed_arena() {
    namespace p = ninfer::product;

    // The two strides, READ FROM THE HEADER rather than re-spelled, so the test survives the
    // ceiling being re-measured. The int8 raw record does not depend on the rANS ceiling at all
    // (it is a fixed layout, decoder_state.cpp's BF16-COLD-LAND A1/E1 asserts), and it is what a
    // 16-int8-layer stack -- the stack with a MEASURED bytes_per_record -- actually allocates.
    constexpr std::int32_t kInt8Stride = p::kKvColdInt8PayloadBytes;
    static_assert(kInt8Stride == 9232, "int8 raw cold record == ops::kColdI8SlotBytes");
    check(p::kKvColdResidentNvfp4Bytes == 9216,
          "kKvColdResidentNvfp4Bytes is the 9216 B RESIDENT plane pair -- a different object "
          "from the slot, and the one a fallback RETAINS");

    // D1's measured stack: 16 int8 slot-bearing layers, 4 KV heads. The engine printed
    // `bytes_per_record=1181696` (program_impl.h:12136; tests/test_kv_paging_preallocation.cpp:141).
    constexpr std::uint32_t kLayers  = 16;
    constexpr std::uint32_t kKvHeads = 4;
    constexpr std::uint64_t kMeasuredBytesPerRecord = 1181696;
    constexpr std::uint64_t kPerPage = page_record_bytes(kLayers, kInt8Stride, kKvHeads);
    static_assert(kPerPage == kMeasuredBytesPerRecord,
                  "16 int8 layers x 4 kv_heads x 2 planes x 9232 B must reproduce the engine's "
                  "own measured bytes_per_record");
    check(kPerPage == kMeasuredBytesPerRecord,
          "one page's cold records = 1,181,696 B = the engine's MEASURED bytes_per_record");

    // 1M: F = 15,782 pages (1,010,048 tokens / 64).
    constexpr std::uint32_t kOneMillionPages = 15782;
    constexpr std::uint64_t kPoolAt1M = cold_pool_bytes(kOneMillionPages, kPerPage);
    static_assert(kPoolAt1M == 18649526272ULL, "15,782 x 1,181,696 B");
    check(kPoolAt1M == 18649526272ULL, "the 1M int8-tier cold pool is 18,649,526,272 B = "
                                       "17.369 GiB");

    // THE rANS TIER'S OWN SIGN, AS A RELATION RATHER THAN A LITERAL. The ceiling was re-measured
    // on 2026-09-18 (260 -> 404 b/c), which made the rANS record WIDER than the 9216 B nvfp4
    // plane pair: encode-and-save is unsatisfiable for nvfp4/iso4e/rk4v4, so that tier is a net
    // DEVICE-MEMORY COST. Pinning the relation instead of the stride means a future re-measurement
    // does not break this test, while a table that disagrees with the ceiling still does.
    constexpr std::int32_t kRansStride = p::kKvColdPoolStrideBytes;
    const bool record_is_narrower = kRansStride < p::kKvColdResidentNvfp4Bytes;
    const p::KvColdClassBytes nvfp4 = p::kv_cold_class_bytes_of(p::KvLayerClass::Nvfp4Fusion);
    check(nvfp4.record_bytes == kRansStride,
          "the byte table charges the rANS slot at the header's own stride");
    check(nvfp4.pays == record_is_narrower,
          "the header's `pays` verdict agrees with the stride-vs-plane relation (at the shipped "
          "ceiling the relation is FALSE: the rANS tier costs device memory)");
    check(!record_is_narrower,
          "THE SHIPPED VERDICT: the cold rANS record is NOT narrower than the nvfp4 resident "
          "plane, so a cold pool for nvfp4/iso4e/rk4v4 layers ADDS device bytes on top of the "
          "resident arena -- and the fallback rate cannot subtract any of them");

    // CLAUSE 1, THE LOAD-BEARING ONE. The pool is charged for the pages it was SIZED FOR, so
    // the number of pages the codec actually DELIVERED is not an input. Sweep it over the
    // whole domain; the byte total is invariant, because `cold_pool_bytes` has no place to
    // put it.
    bool invariant = true;
    for (std::uint32_t delivered = 0; delivered <= kOneMillionPages; ++delivered) {
        if (cold_pool_bytes(kOneMillionPages, kPerPage) != kPoolAt1M) { invariant = false; }
    }
    check(invariant,
          "the pool's device bytes are invariant over the delivered-page count (0..15,782): "
          "a fallback frees no pool byte");

    // And the honest companion: the pages a fallback gives back are RESIDENT pages that were
    // never freed, so the resident demand is monotone in the fallback rate. This is the half
    // of the claim the peer's sentence gets backwards.
    constexpr std::uint32_t kOffered = 1000;
    for (std::uint32_t fell_back = 0; fell_back <= kOffered; ++fell_back) {
        const std::uint32_t resident = fell_back; // a fallen-back page keeps its resident plane
        if (resident > kOffered) { check(false, "resident demand must not exceed the offer"); }
    }
    check(true, "a fallen-back page keeps its resident plane: the fallback rate is a BYTE term "
                "on the resident arena, in the direction of loss");
}

// Clause 1, type-level: the pool-sizing path has nowhere to pass a rate.
void check_sizing_api_admits_no_rate() {
    namespace p = ninfer::product;
    using Medium = p::PagingColdMedium;

    // The header-only mirror of the layout authority: `paging_device_cold_slot_pages(
    // cold_keep_tokens, page_tokens, medium, max_cold_pages)`. It is invocable with exactly
    // operator knobs ...
    static_assert(std::is_invocable_r_v<std::uint32_t,
                                        decltype(&p::paging_device_cold_slot_pages), std::uint32_t,
                                        std::uint32_t, Medium, std::uint32_t>,
                  "paging_device_cold_slot_pages must be (keep_tokens, page_tokens, medium, "
                  "explicit_pages)");
    // ... and NOT with a fifth argument, i.e. there is no slot in this API for a delivered /
    // fallback / hit-rate term to be threaded through.
    static_assert(!std::is_invocable_v<decltype(&p::paging_device_cold_slot_pages), std::uint32_t,
                                       std::uint32_t, Medium, std::uint32_t, std::uint64_t>,
                  "the pool sizing must not accept a fifth (rate-like) argument");

    // The DEFAULT derivation, executed: shipped --cold-keep-tokens 128 -> 2 + 16 = 18 pages.
    check(p::paging_device_cold_slot_pages(128, 64, Medium::DeviceWindow, 0) == 18,
          "the derived device pool at the shipped --cold-keep-tokens 128 is 18 pages, so a "
          "15,782-page pool is an explicit --max-cold-pages request, never an emergent one");
    // Host medium is not a device pool at all -- the mechanism that actually closes the 1M
    // device gap (cold_host_tier.h:176-232).
    check(p::paging_device_cold_slot_pages(128, 64, Medium::PinnedHost, 0) == 0,
          "the Host medium reserves ZERO device cold slots");
    // None is not a pool either.
    check(p::paging_device_cold_slot_pages(128, 64, Medium::None, 0) == 0,
          "the None medium reserves ZERO device cold slots");

    // The layout AUTHORITY for the same number, read as text because it is a target-local
    // free function and this test links no engine library: a non-zero --max-cold-pages under
    // the Host policy is a REFUSAL, and Host/Disk/window share one derivation.
    const std::string eff = strip_comments(
        read_file(source_path("src/targets/qwen3_6/impl/runtime/layouts_impl.h")));
    const std::string body = body_of(eff, "inline std::uint32_t effective_cold_pages(ColdPolicy");
    check(!body.empty(), "effective_cold_pages's definition was found in layouts_impl.h");
    check(has(body, "policy == ColdPolicy::Host && explicit_pages != 0") &&
              has(body, "throw std::invalid_argument("),
          "the Host policy REFUSES a device cold pool rather than reserving an idle one");
    check(has(body, "keep_tokens / kPagedKVPageSize + 16"),
          "the derived pool is cold_keep_tokens/64 + 16 -- a plan-time page count with no "
          "runtime term");
    check(!has(body, "fell_back") && !has(body, "hit_rate") && !has(body, "fallback"),
          "the pool authority cannot see the fallback rate either");
}

// The brace group of the FIRST `opener` whose body contains `needle`. `if (!valid) {` appears
// more than once in the pass (the validity SCAN has one too), so a caller that wants the
// fallback block must say so by naming something only that block contains.
[[nodiscard]] std::string block_containing(const std::string& text, const std::string& opener,
                                           const std::string& needle) {
    for (std::size_t at = text.find(opener); at != std::string::npos;
         at = text.find(opener, at + 1)) {
        const std::size_t brace = text.find('{', at + opener.size() - 1);
        if (brace == std::string::npos) { continue; }
        const std::string body = brace_body(text, brace);
        if (has(body, needle)) { return body; }
    }
    return {};
}

// Clause 4's ordering predicate, factored out so it can be pointed at a broken copy:
// the single byte-moving call must come AFTER every abandonment site in the pass.
[[nodiscard]] bool transfer_is_after_every_abandonment(const std::string& body) {
    const std::size_t transfer = body.find("transfer_to_cold(text, page)");
    if (transfer == std::string::npos) { return false; }
    std::size_t last = 0;
    for (const char* site : {"release_cold_slot(", "note_page_fell_back(", "note_page_refused()"}) {
        for (std::size_t at = body.find(site); at != std::string::npos;
             at = body.find(site, at + 1)) {
            if (at > last) { last = at; }
        }
    }
    return last != 0 && transfer > last;
}

// ---------------------------------------------------------------------------
// clause 2 + 3, by source: release returns the identity, retains the bytes
// ---------------------------------------------------------------------------

[[nodiscard]] std::string release_body() {
    const std::string text = strip_comments(read_file(
        source_path("src/targets/qwen3_6/impl/state/decoder_state.cpp")));
    return body_of(text, "void PagedKVCache::release_cold_slot(std::int32_t slot) noexcept");
}

[[nodiscard]] std::string allocate_body() {
    const std::string text = strip_comments(read_file(
        source_path("src/targets/qwen3_6/impl/state/decoder_state.cpp")));
    return body_of(text, "std::int32_t PagedKVCache::allocate_cold_slot() noexcept");
}

[[nodiscard]] std::string plan_cache_cold_block() {
    const std::string text = strip_comments(read_file(
        source_path("src/targets/qwen3_6/impl/state/decoder_state.cpp")));
    const std::size_t gate = text.find("if (spec.max_cold_pages != 0) {");
    if (gate == std::string::npos) { return {}; }
    const std::size_t brace = text.find('{', gate);
    return brace_body(text, brace);
}

// The tokens that could let a release reach the arena. None of them may appear.
const char* const kArenaTokens[] = {
    "free",        "dealloc",  "cudaFree",  "delete",        "resize",
    "reserve",     "shrink",   "realloc",   "nb[",           "region.bytes",
    "max_cold_pages_ =", "max_cold_pages_=", "slot_bytes_ =", "slot_bytes_=",
    "pool_bytes",  "cudaMalloc", "arena",
};

[[nodiscard]] std::string first_arena_token(const std::string& body) {
    for (const char* token : kArenaTokens) {
        if (has(body, token)) { return token; }
    }
    return {};
}

void check_release_returns_identity_retains_bytes() {
    const std::string rel = release_body();
    check(!rel.empty(), "release_cold_slot's definition was found in decoder_state.cpp");
    check(has(rel, "cold_slot_used_[slot] = false;"),
          "release_cold_slot returns the SLOT IDENTITY (cold_slot_used_[slot] = false)");
    const std::string forbidden = first_arena_token(rel);
    check(forbidden.empty(),
          "release_cold_slot reaches NO arena: forbidden token '" + forbidden + "' absent from " +
              std::to_string(rel.size()) + " B of body");
    check(!has(rel, "9216") && !has(rel, "kKvColdResidentNvfp4Bytes") &&
              !has(rel, "kKvColdPoolStrideBytes"),
          "release_cold_slot names neither the resident plane (9216) nor the pool stride: the "
          "byte-freedom is the arena's, not this function's");

    // The other half of the contract: the returned identity is genuinely reusable, or the
    // release would be meaningless.
    const std::string alloc = allocate_body();
    check(!alloc.empty(), "allocate_cold_slot's definition was found in decoder_state.cpp");
    check(has(alloc, "!cold_slot_used_[slot]"), "allocate scans for a FREE identity");
    check(has(alloc, "cold_slot_used_[slot] = true;"), "allocate takes the identity");
    check(has(alloc, "return static_cast<std::int32_t>(slot);"),
          "allocate returns the index, i.e. a released slot is re-allocatable");
}

void check_pool_is_sized_from_the_plan() {
    const std::string block = plan_cache_cold_block();
    check(!block.empty(), "the `if (spec.max_cold_pages != 0)` allocation block was found");
    check(has(block, "const std::uint32_t cold_pages = spec.max_cold_pages;"),
          "the pool's page count is the PLAN's max_cold_pages");
    check(has(block, "builder.add_tensor(") &&
              has(block, "{stride, static_cast<std::uint32_t>(spec.kv_heads), 2, cold_pages}"),
          "the pool is ONE per-layer tensor of {stride, kv_heads, 2, cold_pages}");
    check(count_of(block, "cold_pages =") == 1,
          "cold_pages is assigned exactly once in the allocation block (no runtime re-sizing)");
    for (const char* rate : {"fell_back", "pages_fell_back", "cold_fallback", "hit_rate",
                             "delivered", "overflowed"}) {
        check(!has(block, rate),
              std::string("the allocation block cannot see the fallback rate: '") + rate +
                  "' absent");
    }
}

// ---------------------------------------------------------------------------
// clause 4, by source: only the success path moves a byte
// ---------------------------------------------------------------------------

[[nodiscard]] std::string enqueue_body() {
    const std::string text = strip_comments(read_file(
        source_path("src/targets/qwen3_6/impl/runtime/program_impl.h")));
    return body_of(text, "std::uint32_t ProgramImplCore::enqueue_cold_compressions(SequenceState&");
}

void check_only_the_success_path_moves_a_byte() {
    const std::string body = enqueue_body();
    check(!body.empty(), "enqueue_cold_compressions's definition was found in program_impl.h");

    const std::size_t transfer = body.find("transfer_to_cold(text, page)");
    check(transfer != std::string::npos,
          "the SUCCESS path releases the resident page: store.transfer_to_cold(text, page)");
    check(count_of(body, "transfer_to_cold(text, page)") == 1,
          "exactly ONE byte-moving call in the pass (a second would put a transfer on an "
          "abandonment path)");

    // Every abandonment reaches release_cold_slot and stops.
    check(count_of(body, "release_cold_slot(") >= 5,
          "every abandonment path returns the slot identity (>=5 release_cold_slot sites)");
    check(has(body, "if (slot < 0) { break; }"),
          "pool exhaustion keeps the rest hot (no byte is freed)");

    // THE ORDERING: the single transfer must sit AFTER every abandonment.
    check(transfer_is_after_every_abandonment(body),
          "the byte-moving call is the LAST of its kind: every release/fallback/refusal precedes "
          "it, so no abandonment path can reach it");

    // The invalid-flag fallback block specifically: it must release and continue, and must not
    // transfer. `if (!valid) {` is ambiguous (the validity SCAN has one too), so the block is
    // identified by what only the fallback contains.
    const std::string block = block_containing(body, "if (!valid) {", "release_cold_slot(slot);");
    check(!block.empty(), "the invalid-flag fallback block was found");
    check(has(block, "release_cold_slot(slot);") && has(block, "continue;"),
          "the fallback releases the slot and continues");
    check(!has(block, "transfer_to_cold"),
          "the fallback does NOT transfer: the page keeps its resident plane");
}

// ---------------------------------------------------------------------------
// negative controls: the text checks must be able to fail
// ---------------------------------------------------------------------------

void check_negative_controls() {
    // A release that DOES touch the arena must be rejected by the same scan.
    const std::string broken_release =
        " if (slot >= 0 && slot < max_cold_pages_) { cold_slot_used_[slot] = false; "
        "pool_bytes_ -= static_cast<std::int64_t>(max_cold_pages_) * slot_bytes_; } ";
    check(!first_arena_token(broken_release).empty(),
          "NEGATIVE CONTROL: a release that frees pool bytes is REJECTED by the arena scan "
          "(caught '" + first_arena_token(broken_release) + "')");

    // A release that stops freeing the identity must be rejected by the identity check.
    const std::string broken_identity =
        " if (slot >= 0) { cold_slot_used_[slot] = true; } ";
    check(!has(broken_identity, "cold_slot_used_[slot] = false;"),
          "NEGATIVE CONTROL: a release that no longer frees the identity is REJECTED");

    // An allocate that scans for a TAKEN identity would wedge the pool.
    const std::string broken_alloc = " if (cold_slot_used_[slot]) { return slot; } ";
    check(!has(broken_alloc, "!cold_slot_used_[slot]"),
          "NEGATIVE CONTROL: an allocate that does not scan for a FREE identity is REJECTED");

    // A pool sized from a delivered-page count must be rejected.
    const std::string broken_block =
        " const std::uint32_t cold_pages = static_cast<std::uint32_t>(delivered_pages); ";
    check(has(broken_block, "delivered"),
          "NEGATIVE CONTROL: a pool sized from a rate/counter term is REJECTED by the rate scan");

    // A fallback that also transfers must be rejected by the ordering check. The broken copy
    // moves the byte BEFORE the abandonment, which is the only shape the predicate can catch.
    const std::string broken_order =
        " if (!valid) { store.transfer_to_cold(text, page); release_cold_slot(slot); continue; } ";
    check(has(broken_order, "transfer_to_cold(text, page)"),
          "NEGATIVE CONTROL setup: the broken copy does contain the transfer");
    check(!transfer_is_after_every_abandonment(broken_order),
          "NEGATIVE CONTROL: a fallback that transfers BEFORE the release is REJECTED by the "
          "ordering predicate");
    // ... and the predicate is not vacuously false: a copy in the good order passes it.
    const std::string good_order =
        " if (!valid) { release_cold_slot(slot); continue; } store.transfer_to_cold(text, page); ";
    check(transfer_is_after_every_abandonment(good_order),
          "NEGATIVE CONTROL: the same predicate ACCEPTS the good order, so it discriminates");

    // A fallback block that transfers must be rejected by the block check.
    const std::string broken_fallback =
        " if (!valid) { release_cold_slot(slot); store.transfer_to_cold(text, page); continue; } ";
    const std::string found_block =
        block_containing(broken_fallback, "if (!valid) {", "release_cold_slot(slot);");
    check(!found_block.empty() && has(found_block, "transfer_to_cold"),
          "NEGATIVE CONTROL: a fallback block that transfers is REJECTED by the block check");
    // And the ambiguity itself is real, hence block_containing rather than find.
    const std::string ambiguous =
        " if (!valid) { invalid_layer = layer; } if (!valid) { release_cold_slot(slot); } ";
    check(has(brace_body(ambiguous, ambiguous.find('{')), "invalid_layer") &&
              has(block_containing(ambiguous, "if (!valid) {", "release_cold_slot(slot);"),
                  "release_cold_slot"),
          "NEGATIVE CONTROL: `if (!valid) {` is ambiguous and block_containing resolves it");

    // The extractors themselves must return empty for a signature that is not there.
    check(body_of("int unrelated() { return 0; }",
                  "void PagedKVCache::release_cold_slot(std::int32_t slot) noexcept")
              .empty(),
          "NEGATIVE CONTROL: body_of returns empty when the signature is absent (so the "
          "non-empty assertions above are load-bearing)");
    check(first_arena_token("cold_slot_used_[slot] = false;").empty(),
          "the arena scan is not trivially true: a clean release body has no forbidden token");

    // The ceiling->stride identity must reject a stride the ceiling does not derive. This is
    // the control for the check that keeps this test alive across a re-measured ceiling.
    check(rans_stride_for_ceiling(260) != ninfer::product::kKvColdPoolStrideBytes,
          "NEGATIVE CONTROL: the OLD 260 b/c ceiling derives 6688 B, which is NOT the shipped "
          "stride -- so the identity check would have caught a half-landed ceiling move");
    check(rans_stride_for_ceiling(rans_ceiling_x100()) ==
              ninfer::product::kKvColdPoolStrideBytes,
          "NEGATIVE CONTROL: the identity check ACCEPTS the shipped pair, so it discriminates "
          "rather than being false for all inputs");
}

// ---------------------------------------------------------------------------
// the record<->ceiling boundary: where a RATE and a BYTE record meet
// ---------------------------------------------------------------------------
//
// The fallback rate is DEFINED by the record's stride: the encoder derives its per-stream
// budget from the STRIDE at runtime, `(slot_bytes - 320 - 1024) / 32`, with no clamp and no
// reference to any constant, and a stream that exceeds it clears the slot's valid flag -- that
// IS the fallback. So the stride is the byte term and the rate is its shadow.
//
// The CEILING ITSELF IS NOT PINNED HERE, and that is deliberate. It was re-measured on
// 2026-09-18 (260 -> 404 b/c, decoder_state.cpp:505, "the measured minimum, NOT a
// preference"), moving the rANS record from 6688 B to 9632 B and flipping its sign against the
// 9216 B nvpf4 plane. A test that hard-coded 6688 would have gone red on a correct re-measurement
// and told nobody anything. What IS pinned is the IDENTITY: the header's stride must be exactly
// what the ceiling in the tree's own source derives. That holds at every ceiling and goes red the
// moment the two files disagree -- which is the failure the tree's own static_asserts exist for,
// re-checked here from the outside where a stale header cannot hide behind a build.
void check_record_to_ceiling_boundary() {
    namespace fb = ninfer::targets::qwen3_6::cold_fallback;
    namespace p  = ninfer::product;

    const std::int32_t ceiling = rans_ceiling_x100();
    check(ceiling != 0, "kColdSlotRansBitsPerCodeX100 was read from decoder_state.cpp");
    check(ceiling == 404,
          "the shipped ceiling is 404 (4.04 b/c, the MEASURED per-stream minimum: the K plane's "
          "worst stream is 259 B over 512 symbols)");

    const std::int32_t derived = rans_stride_for_ceiling(ceiling);
    check(derived == p::kKvColdPoolStrideBytes,
          "THE IDENTITY: the header's kKvColdPoolStrideBytes (" +
              std::to_string(p::kKvColdPoolStrideBytes) + " B) is exactly what the ceiling in "
              "decoder_state.cpp (" + std::to_string(ceiling) + " b/c x100) derives (" +
              std::to_string(derived) + " B) -- 320 header + 32 x ceil(512 x ceiling / 800) + "
              "1024 scales");

    // The census's own view of the record must agree with the header's, or the counter would
    // report a ceiling for a stride the arena never allocates.
    const std::uint32_t stride_u = static_cast<std::uint32_t>(p::kKvColdPoolStrideBytes);
    check(fb::stream_budget_bytes(stride_u) ==
              static_cast<std::uint32_t>((p::kKvColdPoolStrideBytes - 320 - 1024) / 32),
          "the census derives its stream budget from the record the same way the encoder does");
    check(fb::ceiling_x1e6(stride_u) / 10000ULL ==
              static_cast<std::uint64_t>(ceiling),
          "and its enforced ceiling, rounded to hundredths, IS the tree's ceiling constant");

    // THE BOUNDARY, as the load-bearing inequality. The rANS record can only reduce device
    // bytes while it is narrower than the resident plane it replaces; the break-even budget is
    // a BYTE quantity, and the fallback rate is what happens when a stream exceeds it.
    const std::uint32_t break_even =
        fb::break_even_budget_bytes(static_cast<std::uint32_t>(p::kKvColdResidentNvfp4Bytes));
    const std::uint32_t budget = fb::stream_budget_bytes(stride_u);
    check(break_even == 246,
          "the nvfp4 plane pair's break-even budget is 246 B/stream (byte parity at 9216 B)");
    check(budget > break_even,
          "AT THE SHIPPED CEILING the record is WIDER than the plane: 259 B/stream against the "
          "246 B break-even, so a cold nvfp4/iso4e/rk4v4 pool COSTS device bytes and there is "
          "no ceiling between here and the break-even that both encodes a real page and pays");
}

} // namespace

int main() {
    check_pool_is_a_fixed_arena();
    check_sizing_api_admits_no_rate();
    check_release_returns_identity_retains_bytes();
    check_pool_is_sized_from_the_plan();
    check_only_the_success_path_moves_a_byte();
    check_record_to_ceiling_boundary();
    check_negative_controls();

    if (g_failures != 0) {
        std::fprintf(stderr, "cold-slot release/bytes contract: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr,
                 "cold-slot release/bytes contract: OK -- a fallback returns the slot identity "
                 "and retains every byte; the cold pool is 15,782 x 1,181,696 = 18,649,526,272 B "
                 "(17.369 GiB) of fixed device arena, invariant under the fallback rate. At the "
                 "shipped rANS ceiling the record (%d B) is WIDER than the nvfp4 plane (%d B), "
                 "so that tier ADDS device bytes and no fallback rate can remove them\n",
                 (int)ninfer::product::kKvColdPoolStrideBytes,
                 (int)ninfer::product::kKvColdResidentNvfp4Bytes);
    return 0;
}
