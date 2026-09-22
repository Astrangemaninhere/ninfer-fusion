#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
    std::size_t max_tokens  = std::numeric_limits<std::size_t>::max();
};

struct ByteSpan {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct DecodedTokenView {
    std::string_view bytes;
    bool special = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

struct BpeMergeRule {
    int rank   = 0;
    int result = -1;
};

struct BpeMergeEntry {
    std::uint64_t key = 0;
    int rank          = 0;
    int result        = -1;
};

// Merge table of the BPE encoder, open addressed with linear probing.
//
// Every adjacent symbol pair of every word is looked up here, and the lookup dominates
// Tokenizer::encode, so the table is stored flat: one power-of-two array of 16-byte entries
// instead of a bucket array plus one heap node per rule. A hit reads a single cache line
// rather than following a pointer chase, and the whole structure is smaller than the
// equivalent std::unordered_map even at a load factor below one half.
//
// The invariant that makes an empty slot recognisable is result < 0. Token ids come from
// parse_token_id, which rejects anything negative, so a stored rule always has result >= 0.
class BpeMergeTable {
public:
    // Sizes the table for rule_count rules. Capacity is the smallest power of two above
    // twice that count, so at least half of the slots stay empty and every probe chain ends.
    void reset(std::size_t rule_count) {
        std::size_t capacity = 1;
        while (capacity < rule_count * 2U) { capacity <<= 1U; }
        slots_.assign(capacity, BpeMergeEntry{});
        mask_ = capacity - 1U;
    }

    // Returns false when the key is already present, which is how the loader detects a
    // duplicate pair in model.merges.
    bool insert(std::uint64_t key, BpeMergeRule rule) {
        std::uint64_t index = mix(key) & mask_;
        while (slots_[index].result >= 0) {
            if (slots_[index].key == key) { return false; }
            index = (index + 1U) & mask_;
        }
        slots_[index] = BpeMergeEntry{.key = key, .rank = rule.rank, .result = rule.result};
        return true;
    }

    [[nodiscard]] const BpeMergeEntry* find(std::uint64_t key) const noexcept {
        std::uint64_t index = mix(key) & mask_;
        while (true) {
            const BpeMergeEntry& entry = slots_[index];
            if (entry.result < 0) { return nullptr; }
            if (entry.key == key) { return &entry; }
            index = (index + 1U) & mask_;
        }
    }

private:
    // splitmix64 finalizer. A key is two token ids packed into one word, so its low bits
    // alone would cluster badly across a power-of-two slot count.
    static std::uint64_t mix(std::uint64_t key) noexcept {
        key += 0x9e3779b97f4a7c15ULL;
        key = (key ^ (key >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        key = (key ^ (key >> 27U)) * 0x94d049bb133111ebULL;
        return key ^ (key >> 31U);
    }

    // One empty slot by default, so find() on a default-constructed table terminates.
    std::vector<BpeMergeEntry> slots_ = std::vector<BpeMergeEntry>(1);
    std::uint64_t mask_               = 0;
};

struct TokenBoundaryResult {
    // exact_frontier is present when the byte marker is also a boundary in the complete token
    // stream. stable_frontier retains only a normalization- and pre-tokenization-complete prefix.
    std::optional<std::size_t> exact_frontier;
    std::size_t stable_frontier = 0;
};

struct BoundaryEncodedText {
    std::vector<int> input_ids;
    std::vector<TokenBoundaryResult> boundaries;
    // NOTE ON LINE NUMBERS: the `file:line` references in this block and in the declaration of
    // `encode_with_frame_classes` below are PRE-LANDING (HEAD 3944a53). The landing inserts
    // lines into this file and into tokenizer.cpp, so the numbers shift by however many lines
    // were inserted above them; each reference is also given by SYMBOL. Current values:
    //   tokenizer.h BoundaryEncodedText:128  token_classes:158  FrameClassifiedEncode:165
    //   tokenizer.cpp append_normalized_bpe_ids:552  ids.push_back:623  match_token==nullptr:869
    //   tokenizer.cpp push(match_token->id):884  encode_with_frame_classes:941
    //
    // THE FRAME AXIS COLUMN. One class byte per entry of `input_ids`, in lockstep:
    //   Literal         -- the token came from bytes the CALLER supplied
    //   Template        -- the token came from bytes the chat template supplied
    //   TemplateControl -- an added/special token matched inside template bytes
    // (`ninfer::spec::frame_axis::TokenClass`; the values are identical and pinned by a
    // static_assert next to the filler.)
    //
    // WHERE THE CLASS IS BORN, and why this column had to exist: `encode_with_boundaries` knows the
    // class at the instant it pushes each id -- `:873 append_ordinary(...)` pushes ordinary-text ids
    // and `:883 encoded.input_ids.push_back(match_token->id)` pushes an added token's -- and there
    // are exactly those TWO id-push sites in the whole encode path (`append_normalized_bpe_ids`
    // pushes at :622 and is reached only through the first of them). What was missing is a PLACE to
    // put it: this struct had none, `EncodedChat` had none, and `ProcessedInput`'s one per-token byte
    // column is the VISION MODALITY column, which is not a free annotation channel
    // (product/kv_recall_block.h:352-360).
    //
    // EMPTY MEANS "NOT CLASSIFIED", which is a different thing from "every token is Literal" -- the
    // distinction is load-bearing and is drawn by `size() == 0` vs `size() == input_ids.size()`.
    // `encode_with_boundaries` leaves it empty; only `encode_with_frame_classes` fills it.
    std::vector<std::uint8_t> token_classes;
};

// The result of the frame-axis encode. `literal_spans_resolved` / `..unresolved` are reported rather
// than swallowed: an unresolvable span leaves its tokens on the TEMPLATE side, which is the
// conservative direction for the one consumer that matters (a block that may not be re-sent as
// ordinary input) and is never silent.
struct FrameClassifiedEncode {
    BoundaryEncodedText encoded;
    std::uint32_t literal_spans_resolved   = 0;
    std::uint32_t literal_spans_unresolved = 0;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    // THE FRAME-AXIS ENTRY POINT. Same `input_ids` and same `boundaries` as
    // `encode_with_boundaries` for the same (text, options, literal_spans) -- this is asserted, not
    // assumed, and the implementation note says why -- PLUS a per-token class column.
    //
    // It is a SEPARATE entry point and not a new default on the existing one, because the class
    // column costs a second tokenization pass and because the existing entry point's reported
    // frontiers must not move by even one token. OFF BY DEFAULT, opt-in per call.
    FrameClassifiedEncode encode_with_frame_classes(std::string_view text,
                                                   std::span<const std::size_t> byte_boundaries,
                                                   EncodeOptions options = {},
                                                   std::span<const ByteSpan> literal_spans = {}) const;

    BoundaryEncodedText encode_with_boundaries(std::string_view text,
                                               std::span<const std::size_t> byte_boundaries,
                                               EncodeOptions options                   = {},
                                               std::span<const ByteSpan> literal_spans = {}) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    [[nodiscard]] DecodedTokenView decoded_token(int id) const;
    [[nodiscard]] std::string_view decode_token_bytes(int id,
                                                      bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
    std::vector<std::string> decoded_token_bytes_;
    std::vector<bool> valid_token_ids_;
    std::vector<bool> special_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    BpeMergeTable bpe_merge_rules_;
    std::array<int, 256> byte_token_ids_{};
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
