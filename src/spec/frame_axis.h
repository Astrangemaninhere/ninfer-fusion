#pragma once

// src/spec/frame_axis.h -- THE FRAME AXIS: which token positions are template/control, as a column.
//
// ONE LINE. The engine has always known, token by token, whether a token came from the CHAT
// TEMPLATE or from the CALLER'S LITERAL TEXT -- and has always thrown that knowledge away. This
// header is the column that keeps it, in a form that rides inside the existing SumDir row stride
// with NO WIRE CHANGE: `SumDirRow::frame_bits` is written at byte offset 70 of an 80-byte stride
// whose bytes 70..80 were previously neither written nor read (pack_row's last write ended at 70,
// unpack_row's last read ended at 70, kSumDirRowWireBytes == 80).
//
// NOTE ON LINE NUMBERS (added with this landing, and deliberately blunt about it). Every
// `file:line` in this header was read off the tree at HEAD 3944a53 BEFORE the change; each one is
// also identified by SYMBOL, because this landing inserts lines into some of the files it cites and
// a bare number would then point at the wrong line. Current values, re-verified after the landing:
//   tokenizer.cpp   append_normalized_bpe_ids:552  ids.push_back:623  has_internal_boundary:657
//                   independently_normalized != normalized:697  encode_with_boundaries:797
//                   match_token == nullptr:869  push(match_token->id):884
//                   encode_with_frame_classes:941
//   tokenizer.h     BoundaryEncodedText:128  token_classes:158  FrameClassifiedEncode:165
//   processor.cpp   assign_positions:642  encode_rendered_chat:761  exact-frontier throw:814
//   processor.h     ProcessedInput::token_classes:121  EncodedChat::token_classes:141
//   sum_dir.h       SumDirRow:435  kSumDirRowWireBytes:498  static_assert:499  block facts:916
//                   holds[] (the judge's only read of them):968  sum_dir_frame_facts_apply:1028
//                   pack_row frame_bits:1913  unpack_row frame_bits:1936

// WHERE THE CLASS IS BORN (producer; the numbers are the PRE-LANDING ones)
//   chat_template.cpp:68    `void append_template(std::string_view text) { fragment_.text += text; }`
//                           -> these bytes ARE the template; NO span is recorded.
//   chat_template.cpp:70-74 `append_literal(...)` -> `append_literal_span(fragment_.literal_spans,
//                           ByteSpan{begin, ...})` -> these bytes ARE the caller's literal text;
//                           a span IS recorded.
//   chat_template.h:120     `std::vector<ByteSpan> literal_spans;`
//   tokenizer.cpp:868       `if (match_token == nullptr) { ++pos; continue; }` -- this byte is not an
//                           added/special token.
//   tokenizer.cpp:873       `if (!append_ordinary(ordinary_begin, pos)) { return encoded; }` pushes
//                           the ordinary ids; tokenizer.cpp:883
//                           `encoded.input_ids.push_back(match_token->id);` pushes the ADDED token's
//                           id. => at the instant of the push the class is known.
//   tokenizer.h:128-131     `BoundaryEncodedText{ input_ids; boundaries; }`   -- NO class column.
//   processor.h:119-134     `EncodedChat{ input_ids; ... }`                   -- NO class column.
//   processor.h:101-117     `ProcessedInput{ input_ids; token_types; ... }`   -- NO class column.
//   processor.cpp:646       `output.token_types.assign(length, 0);`           -- and the ONE per-token
//                           byte column that DOES exist is the VISION MODALITY column, so it cannot
//                           be borrowed (product/kv_recall_block.h:352-360 pins that verbatim:
//                           non-zero means a Vision chunk is owed).
//
// HOW THIS FILE RECOVERS THE AXIS WITHOUT TOUCHING THE SCANNER
//   The tokenizer ALREADY publishes the answer -- not as a class but as a FRONTIER. A caller who
//   passes a byte offset in `byte_boundaries` gets back a `TokenBoundaryResult{exact_frontier,
//   stable_frontier}` for it (tokenizer.h:121-126; filled at tokenizer.cpp:661-700 for ordinary text
//   and tokenizer.cpp:875-891 for added tokens). The literal-span EDGES are the very same kind of
//   marker, and the prefill caller (`encode_rendered_chat`, processor.cpp:743) already holds
//   `rendered.literal_spans` and already asks for other markers at processor.cpp:749-772. So the
//   class column is a pure function of (literal edges, reported frontiers): it needs NO new scanning
//   logic and NO edit to any line of the tokenizer's loop. That is why this is a spec header with a
//   unit-testable core rather than a patch to the tokenizer's inner loop.
//
// WHAT IT IS NOT
//   * NOT `PreparedPromptData::token_types`. That column's 0 means "ordinary text" and non-zero means
//     "a Vision chunk is owed" (text_prefill_impl.h:349-363, program_impl.h:4443-4447): it is a
//     modality column with a live consumer that THROWS when the chunk is missing, so it is not a
//     free annotation channel. `FrameAxis::classes` is a NEW column whose 0 means LITERAL.
//   * NOT per-token storage of the whole sequence in the row. `FrameBitmap` is per BLOCK (64 tokens),
//     exactly 8 bytes, which is what fits the row's free stride.
//   * NOT a query class. See spec/sum_dir_query_registry.h for the Q<->field registration.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::spec::frame_axis {

// The granularity is the Paged-KV page, and only that (sum_dir.h:136-138 static_asserts it).
inline constexpr std::uint32_t kFrameBlockTokens = 64U;

enum class TokenClass : std::uint8_t {
    Literal         = 0, // bytes the CALLER supplied: user content
    Template        = 1, // bytes the CHAT TEMPLATE supplied: glue, role labels, delimiters
    TemplateControl = 2, // an added/special token matched inside template bytes (e.g. <|im_start|>)
    Media           = 3, // a media placeholder token run (template-supplied, so also frame)
};

inline constexpr std::uint8_t kFrameClassCount = 4U;

[[nodiscard]] constexpr bool frame_class_known(std::uint8_t raw) noexcept {
    return raw < kFrameClassCount;
}

// THE frame predicate: "is this position a template/control/role-boundary position?"
[[nodiscard]] constexpr bool frame_class_is_frame(std::uint8_t raw) noexcept {
    return frame_class_known(raw) && raw != static_cast<std::uint8_t>(TokenClass::Literal);
}

// One block's frame bitmap. Bit j == 1  <=>  token offset j of the block is a frame position.
// == 0 means "no frame position in this block", which is EXACTLY the semantics of a row written by
// a writer that predates this column, so an old file stays readable and means what it always meant.
using FrameBitmap = std::uint64_t;

[[nodiscard]] constexpr FrameBitmap frame_bit(std::size_t index) noexcept {
    return index < kFrameBlockTokens ? (FrameBitmap{1} << index) : FrameBitmap{0};
}

// The mask of the first `tokens` offsets of a block. tokens >= 64 is the full block.
[[nodiscard]] constexpr FrameBitmap frame_token_mask(std::uint32_t tokens) noexcept {
    if (tokens == 0U) { return FrameBitmap{0}; }
    if (tokens >= kFrameBlockTokens) { return ~FrameBitmap{0}; }
    return (FrameBitmap{1} << tokens) - FrameBitmap{1};
}

// The class column of the WHOLE sequence, and one block of it -> that block's bitmap.
[[nodiscard]] inline FrameBitmap frame_bitmap_from_class_column(
    std::span<const std::uint8_t> classes, std::size_t begin, std::uint32_t count) noexcept {
    FrameBitmap bits = 0;
    if (begin >= classes.size()) { return 0; }
    const std::size_t end = std::min(classes.size(), begin + static_cast<std::size_t>(count));
    for (std::size_t index = begin; index < end; ++index) {
        if (frame_class_is_frame(classes[index])) { bits |= frame_bit(index - begin); }
    }
    return bits;
}

// ---------------------------------------------------------------------------
// THE LITERAL -> TOKEN HANDOFF (the only place bytes become token indices)
// ---------------------------------------------------------------------------
//
// A literal token range is the half-open TOKEN interval [begin, end) that one literal byte span
// occupies. `RenderedChat::literal_spans` are BYTE intervals; the tokenizer turns a byte offset into
// a token frontier and nothing else does. So a caller builds these by asking the tokenizer for the
// two edges of each literal span and pairing the answers in order.
struct LiteralTokenRange {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
};

// Pairs two equal-length sequences of token frontiers into literal token ranges, and DROPS a pair
// whose end does not exceed its begin (an empty literal span, or an edge pair that normalization
// collapsed). Ordered and non-overlapping on the way out; the tokenizer has already refused an
// unordered literal-span list (tokenizer.cpp:802-808), so a violation here is a caller bug and is
// reported rather than silently repaired.
[[nodiscard]] inline bool literal_token_ranges_from_frontiers(
    std::span<const std::uint32_t> literal_begin_frontier,
    std::span<const std::uint32_t> literal_end_frontier,
    std::vector<LiteralTokenRange>& out) {
    if (literal_begin_frontier.size() != literal_end_frontier.size()) { return false; }
    out.clear();
    for (std::size_t index = 0; index < literal_begin_frontier.size(); ++index) {
        const std::uint32_t begin = literal_begin_frontier[index];
        const std::uint32_t end   = literal_end_frontier[index];
        if (end < begin) { return false; }
        if (end == begin) { continue; }
        if (!out.empty() && begin < out.back().end) { return false; }
        out.push_back(LiteralTokenRange{.begin = begin, .end = end});
    }
    return true;
}

// The class column over `token_count` tokens: LITERAL inside a literal token range, TEMPLATE
// everywhere else. The complement rule is the one chat_template.cpp:68 implies, because
// `append_template` is the ONLY writer that records nothing while `append_literal` records a span:
// the template's extent is "everything that is not a literal span".
[[nodiscard]] inline std::vector<std::uint8_t> class_column_from_literal_token_ranges(
    std::uint32_t token_count, const std::vector<LiteralTokenRange>& literal) {
    std::vector<std::uint8_t> classes(
        static_cast<std::size_t>(token_count),
        static_cast<std::uint8_t>(TokenClass::Template));
    for (const LiteralTokenRange& range : literal) {
        const std::uint32_t begin = std::min(range.begin, token_count);
        const std::uint32_t end   = std::min(range.end, token_count);
        for (std::uint32_t index = begin; index < end; ++index) {
            classes[index] = static_cast<std::uint8_t>(TokenClass::Literal);
        }
    }
    return classes;
}

// ---------------------------------------------------------------------------
// THE THREE FACTS THE FRAME AXIS CAN DECIDE BY ITSELF (and the honest boundary of each)
// ---------------------------------------------------------------------------
//
// Mapping onto `SumDirBlockFacts` (sum_dir.h:883-890):
//
//   has_control_token    <- frame_bits != 0                         EXACT. A non-zero bit IS a template
//                           or added-token position inside this block, and the row's own comment
//                           (:888) says "an injected control or template token is in it".
//   inside_system_prefix <- frame_bits == frame_token_mask(tokens)  SUFFICIENT, NOT NECESSARY. Every
//                           block wholly inside the system preamble satisfies it, so the verdict
//                           `SystemPrefix` becomes REACHABLE for the first time -- but a block of
//                           pure template glue BETWEEN two user turns satisfies it too, so this can
//                           over-report. The missing input is the template's own message extent
//                           (`RenderedChat::message_boundaries`, chat_template.h:125-128), which the
//                           engine side already carries as `ProcessedInput::message_boundaries`;
//                           narrowing this to the true preamble is a follow-up, not a claim made here.
//   pinned_by_anchor     <- boundary_bits != 0                      EXACT, given the frontiers. The
//                           bits come from frontiers the sequence ALREADY carries
//                           (`EncodedChat::rewrite_execution_frontiers` / `message_boundaries` /
//                           `cache_boundaries`, processor.h:130-133). This is what makes
//                           `pinned_by_anchor` fillable without new engine state: sum_dir.h:887 says
//                           "an anchor / checkpoint frontier sits inside it", and those frontiers are
//                           the frontiers the request already publishes.
struct FrameFacts {
    bool has_control_token    = false;
    bool inside_system_prefix = false;
    bool pinned_by_anchor     = false;
};

[[nodiscard]] inline FrameFacts frame_facts(FrameBitmap frame_bits, FrameBitmap boundary_bits,
                                            std::uint32_t tokens) noexcept {
    FrameFacts facts;
    facts.has_control_token    = frame_bits != 0;
    facts.inside_system_prefix = tokens != 0 && frame_bits == frame_token_mask(tokens);
    facts.pinned_by_anchor     = boundary_bits != 0;
    return facts;
}

// ---------------------------------------------------------------------------
// THE PER-SEQUENCE AXIS
// ---------------------------------------------------------------------------
//
// Two bitmaps per block and no more: 16 bytes/block in RAM, 8 bytes/block in the row (only the frame
// bitmap is persisted, because that is the column the row's free stride affords -- and the boundary
// bitmap is recomputable from the sequence's own frontiers at any time, so it does NOT need to be).
struct FrameAxis {
    std::uint32_t tokens = 0;                  // committed token count of the sequence
    std::vector<FrameBitmap> frame_bits;       // one per block: the r side (SumDirRow::frame_bits)
    std::vector<FrameBitmap> boundary_bits;    // one per block: anchor/message/cache frontiers
    std::vector<std::uint8_t> classes;         // the producer's per-token column (may be empty)

    [[nodiscard]] std::uint32_t block_count() const noexcept {
        return (tokens + kFrameBlockTokens - 1U) / kFrameBlockTokens;
    }
    [[nodiscard]] bool shaped() const noexcept {
        return frame_bits.size() == block_count() && boundary_bits.size() == block_count();
    }
    [[nodiscard]] FrameBitmap block_frame_bits(std::uint32_t block) const noexcept {
        return block < frame_bits.size() ? frame_bits[block] : FrameBitmap{0};
    }
    [[nodiscard]] FrameBitmap block_boundary_bits(std::uint32_t block) const noexcept {
        return block < boundary_bits.size() ? boundary_bits[block] : FrameBitmap{0};
    }
    // The token offset range a block owns, clipped to the sequence.
    [[nodiscard]] std::uint32_t block_tokens(std::uint32_t block) const noexcept {
        const std::uint64_t begin = static_cast<std::uint64_t>(block) * kFrameBlockTokens;
        if (begin >= tokens) { return 0U; }
        const std::uint64_t end = std::min<std::uint64_t>(begin + kFrameBlockTokens, tokens);
        return static_cast<std::uint32_t>(end - begin);
    }
};

[[nodiscard]] inline FrameAxis frame_axis_from_class_column(std::uint32_t tokens,
                                                            std::span<const std::uint8_t> classes) {
    FrameAxis axis;
    axis.tokens  = tokens;
    axis.classes.assign(classes.begin(), classes.end());
    axis.frame_bits.assign(axis.block_count(), FrameBitmap{0});
    axis.boundary_bits.assign(axis.block_count(), FrameBitmap{0});
    for (std::uint32_t block = 0; block < axis.block_count(); ++block) {
        axis.frame_bits[block] = frame_bitmap_from_class_column(
            classes, static_cast<std::size_t>(block) * kFrameBlockTokens, kFrameBlockTokens);
    }
    return axis;
}

// A frontier that sits STRICTLY INSIDE a block marks that block. A frontier on a block edge
// (frontier % 64 == 0) is the boundary BETWEEN two blocks and belongs to neither, which is what
// keeps "the anchor sits inside it" (sum_dir.h:887) from claiming the block that merely starts at
// the anchor.
[[nodiscard]] inline bool frame_axis_add_boundary(FrameAxis& axis, std::uint32_t frontier) noexcept {
    if (frontier == 0U || frontier >= axis.tokens) { return false; }
    const std::uint32_t block = frontier / kFrameBlockTokens;
    const std::uint32_t index = frontier % kFrameBlockTokens;
    if (index == 0U || block >= axis.boundary_bits.size()) { return false; }
    axis.boundary_bits[block] |= frame_bit(index);
    return true;
}

// The classes the axis would need to be complete. A caller that cannot supply them must say so
// rather than pass an empty column, because an empty column would read as "no frame position at
// all" -- which is a legitimate value but NOT a legitimate ABSENCE. The two are kept apart by
// `FrameAxis::shaped()` (the bitmaps) and by the explicit status below.
enum class FrameAxisStatus : std::uint8_t {
    Complete      = 0, // classes present, bitmaps match the token count
    NoClassColumn = 1, // no producer column: frame_bits are all zero and mean "unknown"
    ShapeMismatch = 2, // classes.size() != tokens
};

[[nodiscard]] inline FrameAxisStatus frame_axis_status(const FrameAxis& axis) noexcept {
    if (!axis.shaped()) { return FrameAxisStatus::ShapeMismatch; }
    if (axis.classes.empty()) { return FrameAxisStatus::NoClassColumn; }
    if (axis.classes.size() != axis.tokens) { return FrameAxisStatus::ShapeMismatch; }
    return FrameAxisStatus::Complete;
}

// Builds the axis and refuses to hand back a half-filled one: the only two honest outcomes are
// "complete" and "declared absent".
[[nodiscard]] inline FrameAxis frame_axis_classified(std::uint32_t tokens,
                                                     std::span<const std::uint8_t> classes) {
    if (classes.size() != tokens) {
        FrameAxis axis;
        axis.tokens = tokens;
        axis.frame_bits.assign(axis.block_count(), FrameBitmap{0});
        axis.boundary_bits.assign(axis.block_count(), FrameBitmap{0});
        return axis; // NoClassColumn: bitmaps all zero, `status()` says so
    }
    return frame_axis_from_class_column(tokens, classes);
}

// The whole sequence's boundary bits from the frontiers the request already carries.
[[nodiscard]] inline FrameAxis frame_axis_with_boundaries(
    std::uint32_t tokens, std::span<const std::uint8_t> classes,
    std::span<const std::uint32_t> frontiers) {
    FrameAxis axis = frame_axis_classified(tokens, classes);
    for (const std::uint32_t frontier : frontiers) { (void)frame_axis_add_boundary(axis, frontier); }
    return axis;
}

} // namespace ninfer::spec::frame_axis
