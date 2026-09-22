#pragma once
// src/spec/sum_dir_sign_key.h -- THE SECOND CANDIDATE SOURCE: the K sign layer, WIRED.
//
// WHAT THIS FILE IS, AND WHAT IT IS NOT
// -------------------------------------
// `dl/fusiondesign/REPORT.md` sec.4.1 proposed a header named `sum_dir_sign_key.h` carrying a
// `SignSketch`: 256 bits, one sign bit per K dimension, taken from bytes the tree ALREADY writes.
// It was never landed (`SignSketch` and `sum_dir_sign_key` are 0 hits tree-wide -- measured,
// dl/signkey/logs/s01_survey.log:66-77). This file is that header, and it is deliberately shaped
// so that landing it CANNOT move the engine on its own: it is host-only, std-only, and it owns no
// call site. Its one consumer is `install_index_recall_provider` in program_impl.h, which is
// already inside `ensure_sequence_kv_mapped_for_round`'s body -- the prefill gate's own discipline
// (a same-position step, not a second position).
//
// THE MATERIAL IT CARRIES IS THE PLATE'S OWN SIGN FIELD, GATHERED, NOT RE-ENCODED.
// The sign field is defined once, in the e8 lattice codec's byte layout:
//     src/ops/kernel/e8_lattice_codec.cuh:424-433
//       W2, 2 bytes / 8 dims: bits 0..7  = stage-1 table index
//                             bits 8..14 = 7 sign bits
//                             bit  15    = the +-1/4 shift bit
//       W3, 3 bytes / 8 dims: bits 0..15 = the W2 word above
//                             bits 16..23 = the stage-2 table index
//     and the packing line itself, src/ops/kernel/e8_lattice_codec.cuh:437:
//       out[1] = static_cast<std::uint8_t>((c.signs & 0x7fu) | ((c.shift & 1u) << 7));
// => the sign field is BYTE 1 of every 2- or 3-byte group, at BOTH widths. A 256-wide row is 32
// such groups, so the sketch is exactly 32 bytes: the plate's own byte 1 of each group, gathered
// in order. `sum_dir_sign_sketch_from_plate_codes` below is therefore a GATHER, not a second codec
// -- there is one definition of the layout and this file calls its stride rule, it does not
// restate a packing.
//
// ⛔ THE DOMAIN, AND WHY THIS CAN NEVER NAME A BLOCK
// `src/ops/kernel/e8_lattice_kv_plane.cuh:216-217` states it in one line each: the writer takes
// RAW K (`kE8KvLatticeInputIsNatural = true`) and the reader emits ROTATED K
// (`kE8KvLatticeOutputIsRotated = true`). K is stored rotated, the rotation carries position, and
// this tree has no inverse rotation (`dl/kvmine/REPORT.md:32-52`: the whole-tree grep for
// rerotate|re_rope|remove_rope|unrope|de_rope|derope|inv_rope|rope_inverse returns empty). So the
// sign field is a LOCATOR, never an IDENTITY, and `src/spec/sum_dir.h:90` forbids anything
// positional from entering `block_identity`. This file asserts the domain in the COMPILER and
// admits exactly ONE source, so that the one tempting shortcut (use the 0-byte sign field as the
// key) cannot be taken without deleting an assertion that names itself.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::spec::sum_dir {

// ---------------------------------------------------------------------------
// THE SKETCH: 256 bits = 32 B, one sign byte per 8 K dimensions.
// ---------------------------------------------------------------------------
inline constexpr std::uint32_t kSignSketchBits          = 256U;
inline constexpr std::uint32_t kSignSketchBytes         = kSignSketchBits / 8U;   // 32
inline constexpr std::uint32_t kSignSketchGroupDims     = 8U;                     // the codec's 8
inline constexpr std::uint32_t kSignSketchGroups        = kSignSketchBits / kSignSketchGroupDims; // 32
static_assert(kSignSketchBytes == 32U, "256 bits is 32 bytes");
static_assert(kSignSketchGroups == kSignSketchBytes,
              "one group per 8 dims at 256 dims is 32 groups and the sign field is one byte per "
              "group -- that identity is WHY the sketch is exactly as wide as the sign field");

// The plate's own code stride, restated from `kE8KvLatticeCodeBytesPer8<WBits>`
// (e8_lattice_kv_plane.cuh:177) and from `e8_kv_code_bytes_per_8` (product/kv_e8_width.h:271).
// A CUDA header cannot be included here (this file is host-only and must stay includable by a
// plain host TU), so the rule is restated AND pinned below; the two definitions are the same
// expression, `(8 * bits + 7) / 8`.
[[nodiscard]] constexpr std::uint32_t sum_dir_sign_key_code_bytes_per_8(
    std::uint32_t wbits) noexcept {
    return (8U * wbits + 7U) / 8U;
}
static_assert(sum_dir_sign_key_code_bytes_per_8(2U) == 2U, "W2's plate is 2 bytes / 8 dims");
static_assert(sum_dir_sign_key_code_bytes_per_8(3U) == 3U, "W3's plate is 3 bytes / 8 dims");
// The sign field's own offset inside a group: byte 1, at both widths, because bits 8..15 of the
// little-endian group word are the 7 sign bits plus the shift bit (codec layout above).
inline constexpr std::uint32_t kSignSketchSignByteInGroup = 1U;

struct SignSketch {
    std::array<std::uint8_t, kSignSketchBytes> bits{};
};

// THE PRODUCER (a gather, not a codec). `code` is a row of packed plate codewords in the codec's
// own byte order; `bytes` must cover `kSignSketchGroups` groups at this width.
[[nodiscard]] inline bool sum_dir_sign_sketch_from_plate_codes(
    const std::uint8_t* code, std::size_t bytes, std::uint32_t wbits,
    SignSketch& out) noexcept {
    out = SignSketch{};
    if (code == nullptr) { return false; }
    if (wbits != 2U && wbits != 3U) { return false; }
    const std::size_t stride = sum_dir_sign_key_code_bytes_per_8(wbits);
    if (bytes < static_cast<std::size_t>(kSignSketchGroups) * stride) { return false; }
    for (std::uint32_t g = 0; g < kSignSketchGroups; ++g) {
        out.bits[g] = code[static_cast<std::size_t>(g) * stride + kSignSketchSignByteInGroup];
    }
    return true;
}

// THE DOMAIN CONTRACT, in the idiom `src/spec/recall_identity.h:595` already uses for the same
// class of fact (`kEngineRollingDigestIsPositional`).
inline constexpr bool kSignSketchIsPositional = true;
static_assert(kSignSketchIsPositional,
              "the sign field sits on ROTATED K (kE8KvLatticeOutputIsRotated, "
              "e8_lattice_kv_plane.cuh:217) and therefore carries POSITION -- it is a LOCATOR "
              "and NOTHING positional may enter block_identity (src/spec/sum_dir.h:90)");

// ---------------------------------------------------------------------------
// SOURCE ADMISSION -- exactly one source, and every refusal names itself.
// Shape follows `sum_dir_query_key.h:86-144` (the producer that already had to say this once).
// ---------------------------------------------------------------------------
enum class SumDirSignSketchSource : std::uint8_t {
    QueryPlate = 0,        // ADMITTED: the plate's own sign field, gathered
    EngineRollingDigest,   // refused: mixes the three MRoPE positions
    ModelHiddenState,      // refused: an activation, not a stored plate
    EmbeddingVector,       // refused: a similarity score is not a sign field
    CallerSuppliedIdentity,// refused: it would make the sketch a second, unfalsifiable key
};

[[nodiscard]] constexpr const char* sum_dir_sign_sketch_source_name(
    SumDirSignSketchSource source) noexcept {
    switch (source) {
    case SumDirSignSketchSource::QueryPlate:             return "query-plate";
    case SumDirSignSketchSource::EngineRollingDigest:    return "engine-rolling-digest";
    case SumDirSignSketchSource::ModelHiddenState:       return "model-hidden-state";
    case SumDirSignSketchSource::EmbeddingVector:        return "embedding-vector";
    case SumDirSignSketchSource::CallerSuppliedIdentity: return "caller-supplied-identity";
    }
    return "unknown";
}

[[nodiscard]] constexpr bool sum_dir_sign_sketch_source_admitted(
    SumDirSignSketchSource source) noexcept {
    return source == SumDirSignSketchSource::QueryPlate;
}

[[nodiscard]] constexpr const char* sum_dir_sign_sketch_source_reason(
    SumDirSignSketchSource source) noexcept {
    switch (source) {
    case SumDirSignSketchSource::QueryPlate:
        return "the plate's own sign field: byte 1 of every 8-dim group "
               "(e8_lattice_codec.cuh:437), gathered";
    case SumDirSignSketchSource::EngineRollingDigest:
        return "refused: mixes the three MRoPE positions, and position is exactly what "
               "sum_dir.h:90 excludes from a block's name";
    case SumDirSignSketchSource::ModelHiddenState:
        return "refused: an activation is not a stored plate, and no hidden state on any runtime "
               "path of this tree is a function of the block that was spilled";
    case SumDirSignSketchSource::EmbeddingVector:
        return "refused: a similarity score may narrow the candidate-set UNION and nothing else, "
               "and a score is not a sign field (sum_dir_query_key.h:37-40)";
    case SumDirSignSketchSource::CallerSuppliedIdentity:
        return "refused: it would make the sketch a second key that no codec can falsify";
    }
    return "refused: unknown";
}

static_assert(sum_dir_sign_sketch_source_admitted(SumDirSignSketchSource::QueryPlate),
              "the plate's own sign field is the ONE admitted source");
static_assert(!sum_dir_sign_sketch_source_admitted(SumDirSignSketchSource::EngineRollingDigest),
              "a rolling digest is positional and may never produce a sketch");
static_assert(!sum_dir_sign_sketch_source_admitted(SumDirSignSketchSource::ModelHiddenState),
              "hidden state is not a stored plate");
static_assert(!sum_dir_sign_sketch_source_admitted(SumDirSignSketchSource::EmbeddingVector),
              "a similarity score may narrow the candidate set and nothing else");
static_assert(!sum_dir_sign_sketch_source_admitted(SumDirSignSketchSource::CallerSuppliedIdentity),
              "no unfalsifiable second key");

// ---------------------------------------------------------------------------
// THE CONSUMER: Hamming over the sketch, top-k in ascending distance.
// A TOTAL order (distance ASC, index ASC) so equal-distance rows cannot be dropped by the mere
// presence of other equal-distance rows -- the ranking discipline sum_dir_reach.h:30-44 uses for
// the lexical anchor, stated the same way for the second source.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::uint32_t sum_dir_sign_sketch_hamming(
    const SignSketch& a, const SignSketch& b) noexcept {
    std::uint32_t distance = 0;
    for (std::uint32_t i = 0; i < kSignSketchBytes; ++i) {
        std::uint8_t x = static_cast<std::uint8_t>(a.bits[i] ^ b.bits[i]);
        while (x != 0U) { distance += static_cast<std::uint32_t>(x & 1U); x = static_cast<std::uint8_t>(x >> 1U); }
    }
    return distance;
}

[[nodiscard]] inline std::vector<std::size_t> sum_dir_sign_sketch_top_k(
    const std::vector<SignSketch>& rows, const SignSketch& query, std::size_t k,
    std::vector<std::uint32_t>* distances_out = nullptr) {
    std::vector<std::pair<std::uint32_t, std::size_t>> ranked;
    ranked.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        ranked.emplace_back(sum_dir_sign_sketch_hamming(rows[i], query), i);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) { return a.first < b.first; }
        return a.second < b.second;
    });
    const std::size_t take = std::min(k, ranked.size());
    std::vector<std::size_t> out;
    out.reserve(take);
    if (distances_out != nullptr) { distances_out->assign(take, 0U); }
    for (std::size_t i = 0; i < take; ++i) {
        out.push_back(ranked[i].second);
        if (distances_out != nullptr) { (*distances_out)[i] = ranked[i].first; }
    }
    return out;
}

// ---------------------------------------------------------------------------
// THE SIDE-CAR COLUMN. Row lockstep with the directory's own rows, exactly like the catalogue
// column (`sum_dir.h:439-447`: summaries_.size() == rows_.size() always). It is a SIDE-CAR and
// not a field of `SumDirRow` because that struct's 80-byte stride has every byte accounted for
// (`sum_dir.h:499-513`, `kSumDirRowWireBytes == 80`) and no wire change is allowed here.
//
// ZERO BYTES WHEN NOT FED, and `fed == false` is a NAMED state, not an empty column that reads as
// "no candidates": the difference between "this source found nothing" and "this source was never
// given anything" is the difference between a silent miss and a refusal, and this tree's whole
// refusal discipline (sum_dir_reach.h:249-251) rests on keeping those apart.
// ---------------------------------------------------------------------------
struct SignKeySidecar {
    std::vector<SignSketch>    sketches;   // row lockstep
    std::vector<std::uint32_t> pages;      // row lockstep: the row's own page
    SignSketch                 query{};    // the query's plate sketch for this round
    bool                       fed = false;
    const char*                source = "none";
};

// THE FEED FORMAT (a host file, one record per line, `#` starts a comment):
//   query <64 hex chars>     -- the query's own plate sign field
//   page  <n> <64 hex chars> -- row page n's plate sign field
// This is a HOST surface on purpose: the engine's live configuration never produces a W2/W3
// plate at all (measured -- the rk3v4/rk2v4 tiers are refused by name at
// product/kv_storage_dtype.h:106-120 and layouts_impl.h:129-132, the reader has no caller
// (causal_softmax_attention.cpp:84-95), and every leg runs `--kv-layer-storage 0-15:nvfp4`), so
// the sign field's canonical material is produced OUTSIDE the engine and handed in by name.
[[nodiscard]] inline bool sum_dir_sign_key_parse_hex(const std::string& text, SignSketch& out) {
    // Leading blank padding is tolerated: the record kind and the index are separated by spaces,
    // and a caller that slices the line must not have to know how many.
    std::size_t begin = 0;
    while (begin < text.size() && (text[begin] == ' ' || text[begin] == '\t')) { ++begin; }
    const std::string token = text.substr(begin, 64U);
    if (token.size() != 64U) { return false; }
    auto nibble = [](char ch, std::uint8_t& value) {
        if (ch >= '0' && ch <= '9') { value = static_cast<std::uint8_t>(ch - '0'); return true; }
        if (ch >= 'a' && ch <= 'f') { value = static_cast<std::uint8_t>(ch - 'a' + 10); return true; }
        if (ch >= 'A' && ch <= 'F') { value = static_cast<std::uint8_t>(ch - 'A' + 10); return true; }
        return false;
    };
    SignSketch parsed{};
    for (std::uint32_t i = 0; i < kSignSketchBytes; ++i) {
        std::uint8_t hi = 0, lo = 0;
        if (!nibble(token[2U * i], hi) || !nibble(token[2U * i + 1U], lo)) { return false; }
        parsed.bits[i] = static_cast<std::uint8_t>((hi << 4U) | lo);
    }
    out = parsed;
    return true;
}

namespace detail {
inline void sum_dir_sign_key_trim(std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && (text[begin] == ' ' || text[begin] == '\t')) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && (text[end - 1U] == ' ' || text[end - 1U] == '\t' ||
                           text[end - 1U] == '\r' || text[end - 1U] == '\n')) { --end; }
    text = text.substr(begin, end - begin);
}
} // namespace detail

[[nodiscard]] inline bool sum_dir_sign_key_sidecar_read_feed(
    SignKeySidecar& out, const char* path, std::string* why) {
    out = SignKeySidecar{};
    std::FILE* file = std::fopen(path, "rb");
    if (file == nullptr) {
        if (why != nullptr) { *why = "cannot open feed"; }
        return false;
    }
    std::string line;
    std::uint32_t rows = 0;
    bool have_query = false;
    int ch = 0;
    for (;;) {
        ch = std::fgetc(file);
        const bool eof = (ch == EOF);
        if (!eof && ch != '\n') { line.push_back(static_cast<char>(ch)); continue; }
        detail::sum_dir_sign_key_trim(line);
        if (!line.empty() && line[0] != '#') {
            if (line.rfind("query", 0U) == 0U) {
                // `query <64 hex>`. The kind token is 5 chars and the sketch starts after the
                // space that FOLLOWS it -- not at offset 5, which is that space.
                const std::size_t space = line.find(' ', 5U);
                if (space == std::string::npos) {
                    if (why != nullptr) { *why = "bad query record: no sketch after the kind"; }
                    std::fclose(file);
                    return false;
                }
                SignSketch sketch{};
                if (!sum_dir_sign_key_parse_hex(line.substr(space + 1U), sketch)) {
                    if (why != nullptr) { *why = "bad query record"; }
                    std::fclose(file);
                    return false;
                }
                out.query = sketch;
                have_query = true;
            } else if (line.rfind("page", 0U) == 0U) {
                // `page <index> <64 hex>`. The index may be blank-padded, so the sketch starts
                // after the space that FOLLOWS the index -- not after the first space.
                const std::size_t first = line.find(' ', 4U);
                if (first == std::string::npos) {
                    if (why != nullptr) { *why = "bad page record: no page index"; }
                    std::fclose(file);
                    return false;
                }
                const std::size_t second = line.find(' ', first + 1U);
                if (second == std::string::npos) {
                    if (why != nullptr) { *why = "bad page record: no sketch after the index"; }
                    std::fclose(file);
                    return false;
                }
                const unsigned long page = std::strtoul(line.c_str() + first + 1U, nullptr, 10);
                SignSketch sketch{};
                if (!sum_dir_sign_key_parse_hex(line.substr(second + 1U), sketch)) {
                    if (why != nullptr) { *why = "bad page record: bad sketch"; }
                    std::fclose(file);
                    return false;
                }
                out.pages.push_back(static_cast<std::uint32_t>(page));
                out.sketches.push_back(sketch);
                ++rows;
            } else {
                if (why != nullptr) { *why = "unknown record kind"; }
                std::fclose(file);
                return false;
            }
        }
        line.clear();
        if (eof) { break; }
    }
    std::fclose(file);
    if (!have_query || rows == 0U) {
        if (why != nullptr) { *why = have_query ? "no rows" : "no query record"; }
        return false;
    }
    out.fed = true;
    out.source = "host-feed";
    return true;
}

// THE SECOND SOURCE'S OWN ANSWER. It returns PAGES, never bytes and never a span, so the one
// place a span is ever built stays `reach_request_for_term` in program_impl.h. A page the query
// matches at Hamming distance 0 is not a stronger claim than any other: the sketch is a LOCATOR,
// so all admitted candidates are equally "maybe", and the caller's guard then applies the
// same-page and already-recalled exclusions it already applies to the lexical channel.
[[nodiscard]] inline std::vector<std::uint32_t> sum_dir_sign_key_second_source_pages(
    const SignKeySidecar& sidecar, std::uint32_t fanout_cap, std::size_t* matched_out = nullptr,
    std::uint32_t* best_distance_out = nullptr) {
    std::vector<std::uint32_t> pages;
    if (matched_out != nullptr) { *matched_out = 0U; }
    if (best_distance_out != nullptr) { *best_distance_out = 0U; }
    if (!sidecar.fed || sidecar.sketches.empty() || fanout_cap == 0U) { return pages; }
    std::vector<std::uint32_t> distances;
    const std::vector<std::size_t> top =
        sum_dir_sign_sketch_top_k(sidecar.sketches, sidecar.query, fanout_cap, &distances);
    pages.reserve(top.size());
    for (const std::size_t row : top) {
        pages.push_back(row < sidecar.pages.size() ? sidecar.pages[row] : 0U);
    }
    if (matched_out != nullptr) { *matched_out = pages.size(); }
    if (best_distance_out != nullptr && !distances.empty()) {
        *best_distance_out = distances.front();
    }
    return pages;
}

// THE NAMED LINE. `blocked` is non-null exactly when the source is armed but has no material --
// the state this line exists to make countable, because "the second source produced nothing" and
// "the second source was never fed" must never read the same.
[[nodiscard]] inline std::string sum_dir_sign_key_line(
    bool armed, const SignKeySidecar& sidecar, std::size_t matched, std::uint32_t union_pages,
    std::uint32_t best_distance, const char* blocked) {
    std::string out = "[recall-signkey] armed=";
    out += armed ? "1" : "0";
    out += " source=";
    out += sidecar.source;
    out += " fed=";
    out += sidecar.fed ? "1" : "0";
    out += " rows=" + std::to_string(sidecar.sketches.size());
    out += " matched=" + std::to_string(matched);
    out += " union_pages=" + std::to_string(union_pages);
    out += " best_distance=" + std::to_string(best_distance);
    out += " sketch_bytes=" + std::to_string(kSignSketchBytes);
    if (blocked != nullptr) {
        out += " status=refused no-material reason=";
        out += blocked;
    } else if (!armed) {
        out += " status=disarmed (NINFER_RECALL_SIGNKEY unset)";
    } else if (!sidecar.fed) {
        out += " status=refused no-material reason=sidecar not fed";
    } else if (union_pages == 0U) {
        out += " status=no-candidate (the sign channel found no page)";
    } else {
        out += " status=found";
    }
    out += " -- the second candidate SOURCE: its pages join the candidate set and never the "
           "arbiter (program_impl.h:14868-14872). LOCATOR, never IDENTITY (sum_dir.h:90).";
    return out;
}

} // namespace ninfer::spec::sum_dir
