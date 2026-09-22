#pragma once

// Block-level KV ownership encoding for the recall work package (R1).
//
// WHAT THIS IS: the WRITE side of a per-block sidecar that names each cold block
// by its CONTENT, so that "this KV  <->  which spill slot  <->  which text" can be
// re-established after the device cold slot has been recycled and after the
// process that owned it is gone. It is deliberately pure (std-only): no CUDA, no
// engine headers, no artifact -- the same shape as product/kv_cold_tier_budget.h
// and serve/kv_cold_policy.h, both of which are unit-testable with plain g++.
//
// WHY A SIDECAR AND NOT BITS INSIDE THE COLD SLOT. Read this before proposing to
// "just put the id in the header":
//   * The block table's entry is already an encoding: `entry <= -2` means cold and
//     the slot is `-2 - entry` (core/paged_kv_cache.h:19-32). Its READ side is in
//     the attention kernels -- ops/softmax_attention/dense/causal_cache/
//     small_t_i8.cuh:366-377 and small_t_bf16.cuh:227 -- and its WRITE side is
//     program_impl.h:11045-11048 (publish_indices) plus the codec pack kernels.
//     Changing that encoding is an edit to the attention address path.
//   * The cold slot record has NO spare space. Its stride is a function of the
//     page (decoder_state.cpp:406-413 cold_slot_stride_for takes `page_tokens`),
//     and every byte of both codecs is spoken for: the raw slot's 16-byte header
//     carries kColdI8SlotMagic (ops/kernel/cold_i8_kernels.cuh:37) and the rANS
//     slot is 320 B header + 32 x 167 B streams + 1024 B scales
//     (decoder_state.cpp:434-437). Writing an id into either means editing the
//     pack/decode kernels, i.e. the codec, which is the numerically sensitive
//     area this slice must not touch.
//   * A page is the FLOOR of addressable granularity, not a choice: the record
//     holds a whole (page, head, plane) and the kernels index it by
//     `slot * (2 * kv_heads) + head` (small_t_i8.cuh:372-377). There is no
//     sub-page structure to hang an id on.
// So the identity lives beside the bytes, not inside them.
//
// WHAT THE IDENTITY IS, AND WHY IT IS CONTENT-ONLY. A block is recalled by
// re-prefilling its text as NEW input: the KV is regenerated at NEW positions and
// is self-consistent there. The identity therefore must NOT contain the block's
// absolute position, or a recalled block could never match its own record. And it
// must not be `slot`, because a slot is RECYCLED: PagedKVCache::allocate_cold_slot
// hands back the lowest free index (decoder_state.cpp:722-731). So the digest is
// over content, and the POSITION is carried separately as provenance.
//
// The digest is a SHORTLIST, not an authority -- the house already made exactly
// this call for prefix reuse ("This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse",
// impl/runtime/prefix_identity.h:37-38). Here the authoritative comparison is
// recall_block_content_matches(), which re-reads the ledger span; the digest is
// what a directory can index by without holding every token.

// ninfer/types.h is the header that declares TokenId (std::int32_t,
// include/ninfer/types.h:18) and the SharedCandidateEvidence bits this file parks
// its marker against. It is std-only, so this header stays plain-g++ compilable --
// the same include the sibling product/kv_cold_tier_budget.h:42 makes.
#include "ninfer/types.h"

// spec/fnv_convention.h owns the two FNV-1a 64 conventions this tree uses and the
// static_asserts that keep them from being merged or tripled. It is std-only too
// (one <cstdint>), so it costs this header nothing -- and product/ already includes
// across layer directories (product/kv_rowscale_bake.h includes ops/kernel/...).
#include "spec/fnv_convention.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::product {

// The unit this whole file counts in: one logical Paged-KV page.
//
// 64 is not chosen here -- it IS core/paged_kv_cache.h:17 kPagedKVPageSize, the
// block table's addressing unit (ops/kernel/paged_kv_address.cuh:9-16:
// `block_table[position >> 6]`, shift 6 / mask 63). impl/runtime/cold_host_tier.h:
// 63-69 restates the same number for the same reason and static_asserts it against
// kPagedKVPageSize; this header cannot include that (it drags in cuda_runtime_api.h
// via core/paged_kv_cache.h and would stop being a plain-g++ header), so the pin
// lives in tests/test_kv_recall_block.cpp where both can be included. Nothing else
// in this file may assume 64 beyond that one assertion.
inline constexpr std::uint32_t kRecallBlockTokens = 64;

// Which cold codec produced a block's payload. Mirrors product/
// kv_tier_formats.h:155-159 ColdCodec and, in turn, decoder_state.cpp:340
// ColdSlotCodec -- three values, so two bits would hold it, but the enum is kept
// as-is rather than packed: the point of the field is to let a reader decide
// whether the payload is a raw slot (kKvColdInt8PayloadBytes, 9232 B) or an rANS
// slot (kKvColdPoolStrideBytes, 9632 B), not to save bits. Both are pinned in
// product/kv_tier_formats.h (:348 and :329). This said "6688 B rANS" until
// 2026-09-18; 6688 is the retired 2.60-bit record, which the codec does not produce.
enum class RecallBlockCodec : std::uint8_t {
    None,      // no cold codec: the block is not in the cold tier at all
    Int8Raw,   // raw E2M1-nibble slot (cold_slot_stride_for -> 9232 B)
    Nvfp4Rans, // rANS streams plus an uncompressed scale tail (-> 9632 B)
};

[[nodiscard]] inline std::string_view recall_block_codec_name(RecallBlockCodec codec) noexcept {
    switch (codec) {
    case RecallBlockCodec::None: return "none";
    case RecallBlockCodec::Int8Raw: return "int8-raw-slot";
    case RecallBlockCodec::Nvfp4Rans: return "nvfp4-rans-slot";
    }
    return "?";
}

// The band each field's width comes from, so the record cannot quietly become a
// different shape than the engine's own indices:
//   row / row_generation  <-> core/paged_kv_cache.h:326-376 KVExecutionRowHandle
//                             (row_index() is int32, generation is uint32) and the
//                             pool's row_generations_ vector at :417. A recycled
//                             row must not resolve an old record, and the engine
//                             already guards rows exactly this way.
//   logical_page          <-> SequenceState::ColdPageEntry::page, uint32
//                             (impl/runtime/program.h:506-510)
//   slot / file_slot      <-> SequenceState::ColdPageEntry::slot / ::file_slot,
//                             int32 with -1 meaning "none" (impl/runtime/program.h:
//                             496-510)
//   token_begin/count     <-> SequenceState::ledger_frontier / text_kv_valid, both
//                             uint32 (impl/runtime/program.h:453-459). No narrower
//                             invented width: the ledger is a uint32 range.
struct RecallBlockRecord {
    std::int32_t row              = -1;  // block table row = the sequence's KV identity
    std::uint32_t row_generation  = 0;   // staleness guard against a recycled row
    std::uint32_t logical_page    = 0;   // block index within the row (provenance)
    std::uint32_t token_begin     = 0;   // first ledger token of the block (provenance)
    std::uint32_t token_count     = 0;   // != 0, == kRecallBlockTokens for a full block
    std::int32_t slot             = -1;  // device cold slot, -1 when none
    std::int32_t file_slot        = -1;  // spill-file slot, -1 when none
    RecallBlockCodec codec        = RecallBlockCodec::None;
    std::uint32_t page_tokens     = kRecallBlockTokens; // must equal the engine's page
    // Content identity of the block's tokens, position-independent by design.
    std::uint64_t content_digest = 0;
    bool occupied                = false; // false = the slot holds no live record
    // WHICH SHARD WROTE IT. The same axis, and for the same reason, as `shard_rank`/`shard_world`
    // in spec/turn_recall_journal.h's 64-byte RecallRecord: this sidecar names a `file_slot` and a
    // `slot`, and under a KV split by attention head both of those are per-rank addresses whose
    // stride is 1/world_size of the whole page's. A record read by a world other than the one that
    // wrote it therefore resolves to a DIFFERENT region than it names, silently.
    //
    // The fields sit at the END of the struct and default to the identity world, so every record
    // written before this axis existed is a (0, 1) record and reads back as what it was: a
    // single-device record. Appended rather than inserted so no existing aggregate initialisation
    // or member offset moves.
    std::uint32_t shard_rank  = 0;
    std::uint32_t shard_world = 1;
};

// Why a block was not recorded. A refusal is a RETURN VALUE, never a throw: the
// recall sidecar is an observer of the eviction path, and an observer that can
// abort an eviction would turn a bookkeeping bug into a dead server -- the same
// rule product/kv_cold_tier_budget.h:34-37 applies to an unmet byte budget.
enum class RecallBlockWrite : std::uint8_t {
    Recorded,
    NoCodec,        // the block's layers have no cold codec, so nothing was packed
    LedgerTooShort, // the ledger does not cover [token_begin, token_begin + count)
    BadPageTokens,  // page_tokens != kRecallBlockTokens: the engine's page moved
    BadTokenCount,  // token_count == 0 or > kRecallBlockTokens
    BadShard,       // rank/world_size is not a world (rank >= world_size, or world_size == 0)
};

[[nodiscard]] inline std::string_view recall_block_write_name(RecallBlockWrite result) noexcept {
    switch (result) {
    case RecallBlockWrite::Recorded: return "recorded";
    case RecallBlockWrite::NoCodec: return "no-cold-codec";
    case RecallBlockWrite::LedgerTooShort: return "ledger-too-short";
    case RecallBlockWrite::BadPageTokens: return "bad-page-tokens";
    case RecallBlockWrite::BadTokenCount: return "bad-token-count";
    case RecallBlockWrite::BadShard: return "bad-shard";
    }
    return "?";
}

// The token span a logical page covers: [page * 64, page * 64 + 64). This is the
// engine's own arithmetic for "the pages behind the frontier" -- program_impl.h:
// 10794-10798 divides (text_kv_valid - keep) by kPagedKVPageSize the same way --
// and it is the ONLY place this file turns a page index into a token range.
//
// A block may be PARTIAL: the tail page of a sequence covers only the tokens that
// exist. `valid_tokens` is the sequence's committed token count (SequenceState::
// text_kv_valid, impl/runtime/program.h:459); the span is clamped to it. A partial
// block is still a block: its cold slot holds a full page record whose tail rows
// are whatever the pack wrote, which is why token_count is carried rather than
// derived from the page index alone.
[[nodiscard]] inline std::optional<std::pair<std::uint32_t, std::uint32_t>>
recall_block_span(std::uint32_t logical_page, std::uint32_t valid_tokens) noexcept {
    const std::uint64_t begin = static_cast<std::uint64_t>(logical_page) * kRecallBlockTokens;
    if (begin >= valid_tokens) { return std::nullopt; }
    const std::uint64_t end = begin + kRecallBlockTokens;
    const std::uint64_t clamped = end < valid_tokens ? end : valid_tokens;
    return std::make_pair(static_cast<std::uint32_t>(begin),
                          static_cast<std::uint32_t>(clamped - begin));
}

// 64-bit FNV-1a over the block's tokens.
//
// The constant and the algorithm are the PUBLISHED FNV-1a 64 pair, and they now
// come from the one header that owns both FNV conventions in this tree
// (spec/fnv_convention.h): the published pair here, and the engine's slightly
// different two-lane pair used by the prefix/block shortlist. That header records
// which artefact depends on which convention and carries the static_asserts that
// make it impossible to add a third one, or to "tidy" one of the two into the
// other, without a compile error -- which matters because this digest's value is
// already baked into records on disk.
//
// It is chosen because it is a pure fold over a byte stream with no table and no
// state, so the WRITE side (host, at eviction) and the READ side (wherever a
// recall is resolved) can share this one function and cannot drift; a table-driven
// or randomized hash would put a configuration dependency between them, which is
// the failure mode this whole work package is trying to avoid.
//
// COLLISION HORIZON, stated rather than assumed: 64 bits is a shortlist, so at
// 2^32 distinct blocks the birthday bound is already ~4e-4. That is acceptable
// EXACTLY BECAUSE the digest is never the authority -- recall_block_probe()
// re-reads the tokens and reports a digest/token mismatch as a violation. If a
// future read side wants to skip the token comparison, it must widen this to the
// 128-bit form the house already uses for prefix shortlists
// (impl/runtime/prefix_identity.h:40-57, std::array<std::uint64_t, 2>).
[[nodiscard]] inline std::uint64_t recall_block_digest(std::span<const TokenId> tokens) noexcept {
    constexpr std::uint64_t kFnvOffsetBasis = spec::fnv::kFnv1a64OffsetBasis;
    constexpr std::uint64_t kFnvPrime       = spec::fnv::kFnv1a64Prime;
    std::uint64_t hash = kFnvOffsetBasis;
    for (const TokenId token : tokens) {
        // Fold the token id as 4 little-endian bytes. TokenId is int32
        // (include/ninfer/types.h), and ids are validated into
        // [0, TextConfig::token_domain) before a plan is built
        // (impl/runtime/request_plan_impl.h:235-241), so the bytes are a faithful
        // encoding of the id and not of its sign.
        const std::uint32_t value = static_cast<std::uint32_t>(token);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<std::uint64_t>((value >> (8 * byte)) & 0xFFU);
            hash *= kFnvPrime;
        }
    }
    // Bind the length too: the fold above is prefix-extendable, so without this
    // a block and its own prefix would share a digest at different token counts.
    hash ^= static_cast<std::uint64_t>(tokens.size());
    hash *= kFnvPrime;
    return hash;
}

// THE WRITE SIDE. Called where a page becomes cold, with the ledger span it came
// from. `ledger` is SequenceState::ledger -- the exact committed token sequence
// (impl/runtime/program.h:455), kept in lockstep with execution_frontier by the
// invariants at program_impl.h:13401-13403 -- so the text of a cold block is
// already resident and needs no new capture, only a copy of its span.
//
// `codec` is the codec the pack actually used. Passing None is refused rather than
// recorded, mirroring program_impl.h:10934-10936, where a layer whose dtype has no
// cold codec fails the whole page and the page stays hot. Recording a block with
// no payload would create a directory entry that no recall could ever restore.
// The single-device spelling. Kept byte-for-byte as the overload below's identity-world case, so
// every existing call site and every record already on disk is unaffected.
[[nodiscard]] inline RecallBlockWrite recall_note_block(
    RecallBlockRecord& out, std::span<const TokenId> ledger, std::uint32_t page_tokens,
    std::uint32_t logical_page, std::int32_t row, std::uint32_t row_generation,
    std::int32_t slot, std::int32_t file_slot, RecallBlockCodec codec) noexcept;

// The sharded spelling. `world_size == 1` must be the identity world (rst 0, wst 1); a
// sharded world must name a valid rank. A malformed world is refused rather than recorded,
// because a record stamped with a world that cannot exist is one no reader could accept.
[[nodiscard]] inline RecallBlockWrite recall_note_block(
    RecallBlockRecord& out, std::span<const TokenId> ledger, std::uint32_t page_tokens,
    std::uint32_t logical_page, std::int32_t row, std::uint32_t row_generation,
    std::int32_t slot, std::int32_t file_slot, RecallBlockCodec codec, std::uint32_t rank,
    std::uint32_t world_size) noexcept {
    if (world_size == 0 || rank >= world_size) { return RecallBlockWrite::BadShard; }
    const RecallBlockWrite result = recall_note_block(out, ledger, page_tokens, logical_page, row,
                                                      row_generation, slot, file_slot, codec);
    if (result != RecallBlockWrite::Recorded) { return result; }
    out.shard_rank  = world_size == 1U ? 0U : rank;
    out.shard_world = world_size == 1U ? 1U : world_size;
    return result;
}

[[nodiscard]] inline RecallBlockWrite recall_note_block(
    RecallBlockRecord& out, std::span<const TokenId> ledger, std::uint32_t page_tokens,
    std::uint32_t logical_page, std::int32_t row, std::uint32_t row_generation,
    std::int32_t slot, std::int32_t file_slot, RecallBlockCodec codec) noexcept {
    if (codec == RecallBlockCodec::None) { return RecallBlockWrite::NoCodec; }
    if (page_tokens != kRecallBlockTokens) { return RecallBlockWrite::BadPageTokens; }
    const auto span = recall_block_span(logical_page, static_cast<std::uint32_t>(ledger.size()));
    if (!span.has_value()) { return RecallBlockWrite::LedgerTooShort; }
    const std::uint32_t token_begin = span->first;
    const std::uint32_t token_count = span->second;
    if (token_count == 0 || token_count > kRecallBlockTokens) {
        return RecallBlockWrite::BadTokenCount;
    }

    out.row             = row;
    out.row_generation  = row_generation;
    out.logical_page    = logical_page;
    out.token_begin     = token_begin;
    out.token_count     = token_count;
    out.slot            = slot;
    out.file_slot       = file_slot;
    out.codec           = codec;
    out.page_tokens     = page_tokens;
    out.content_digest  = recall_block_digest(ledger.subspan(token_begin, token_count));
    out.occupied        = true;
    return RecallBlockWrite::Recorded;
}

// THE READER'S RULE for the shard axis: empty on acceptance, a reason otherwise. Same shape as
// multi::recall_record_shard_refusal() in core/shard_rank_axis.h and as
// sharded_name_acceptance() in core/shard_plan.h, so the three halves of one contract read alike.
//
// A record written by another world is NOT a format error: it is a valid record whose
// `file_slot` and `slot` are addresses in a different rank's region. Resolving it here would
// read the wrong bytes and report no failure at all, which is the whole reason the check exists.
[[nodiscard]] inline std::string recall_block_shard_refusal(const RecallBlockRecord& record,
                                                            std::uint32_t rank,
                                                            std::uint32_t world_size) {
    if (world_size == 0) { return "this run's world_size is 0"; }
    if (rank >= world_size) {
        return "this run's rank " + std::to_string(rank) + " is outside world_size " +
               std::to_string(world_size);
    }
    if (record.shard_world == 0) { return "the record names world 0, which is not a world"; }
    if (record.shard_rank >= record.shard_world) {
        return "the record names rank " + std::to_string(record.shard_rank) + " of world " +
               std::to_string(record.shard_world) + ", which is out of range";
    }
    if (record.shard_world != world_size || record.shard_rank != rank) {
        return "the record was written by rank " + std::to_string(record.shard_rank) +
               " of world " + std::to_string(record.shard_world) + " but this run is rank " +
               std::to_string(rank) + " of world " + std::to_string(world_size) +
               ": its slot and file_slot are addresses in another shard's region, so resolving "
               "it would read different bytes and report nothing";
    }
    return {};
}

// The authoritative content comparison, used by the probe and by any future read
// side that must not trust the digest alone.
[[nodiscard]] inline bool recall_block_content_matches(const RecallBlockRecord& record,
                                                       std::span<const TokenId> expected) noexcept {
    if (!record.occupied) { return false; }
    if (expected.size() != record.token_count) { return false; }
    return recall_block_digest(expected) == record.content_digest;
}

// THE PROBE. Returns one line per violation, empty when the record is sound --
// the same shape as serve/kv_cold_policy.h:292-333 invariant_violations(), so it
// can be asserted after every write in a test and reported (never thrown) in
// production. `valid_tokens` is the sequence's committed count at probe time.
// The single-device probe: reports every violation EXCEPT the shard axis, which only a caller
// that knows its world can check. Kept as the overload below's identity-world case.
[[nodiscard]] inline std::vector<std::string>
recall_block_probe(const RecallBlockRecord& record, std::span<const TokenId> ledger,
                   std::uint32_t valid_tokens);

// The sharded probe. With (0, 1) -- the identity world -- `recall_block_shard_refusal` accepts a
// (0, 1) record and refuses anything else, so a single-device probe of a sharded record reports
// the shard violation instead of silently accepting a record whose addresses are not its own.
[[nodiscard]] inline std::vector<std::string>
recall_block_probe(const RecallBlockRecord& record, std::span<const TokenId> ledger,
                   std::uint32_t valid_tokens, std::uint32_t rank, std::uint32_t world_size) {
    std::vector<std::string> bad = recall_block_probe(record, ledger, valid_tokens);
    const std::string shard = recall_block_shard_refusal(record, rank, world_size);
    if (!shard.empty()) { bad.push_back(shard); }
    return bad;
}

[[nodiscard]] inline std::vector<std::string>
recall_block_probe(const RecallBlockRecord& record, std::span<const TokenId> ledger,
                   std::uint32_t valid_tokens) {
    std::vector<std::string> bad;
    const auto fail = [&bad](std::string message) { bad.push_back(std::move(message)); };

    if (!record.occupied) {
        fail("empty record probed as live");
        return bad;
    }
    if (record.row < 0) { fail("row is negative"); }
    if (record.page_tokens != kRecallBlockTokens) {
        fail("page_tokens " + std::to_string(record.page_tokens) + " != " +
             std::to_string(kRecallBlockTokens));
    }
    if (record.token_count == 0 || record.token_count > kRecallBlockTokens) {
        fail("token_count " + std::to_string(record.token_count) + " outside (0, " +
             std::to_string(kRecallBlockTokens) + "]");
    }

    // The span must be the one the page index names. This is the check that keeps
    // token_begin from being decoration: a record whose span disagrees with its
    // page is a record whose text would be recalled from the wrong place.
    const auto span = recall_block_span(record.logical_page, valid_tokens);
    if (!span.has_value()) {
        fail("logical_page " + std::to_string(record.logical_page) +
             " lies past valid_tokens " + std::to_string(valid_tokens));
    } else {
        if (span->first != record.token_begin) {
            fail("token_begin " + std::to_string(record.token_begin) + " != page-derived " +
                 std::to_string(span->first));
        }
        if (span->second != record.token_count) {
            fail("token_count " + std::to_string(record.token_count) + " != page-derived " +
                 std::to_string(span->second));
        }
    }

    if (record.token_begin + record.token_count > ledger.size()) {
        fail("span [" + std::to_string(record.token_begin) + ", " +
             std::to_string(record.token_begin + record.token_count) +
             ") exceeds ledger size " + std::to_string(ledger.size()));
    } else {
        // Re-derive the digest from the text. This is the read/write-consistency
        // check the Rk4v4 gate bug argues for: the writer's digest and the reader's
        // recomputation must be the SAME function over the SAME bytes, so the
        // probe recomputes rather than trusting the stored value.
        const std::uint64_t recomputed = recall_block_digest(
            ledger.subspan(record.token_begin, record.token_count));
        if (recomputed != record.content_digest) {
            fail("content_digest does not re-derive from the ledger span");
        }
    }

    return bad;
}

// The recall injection marker bits, parked next to the encoder because they are the
// same contract's second half: a recalled block must reach the model as ORDINARY
// context, so its annotation has to live somewhere the model cannot see.
//
// The channel is PreparedContextCache::opportunities
// (export/ninfer/targets/qwen3_6/prepared_prompt.h:104-120): a frontend -> engine
// annotation list of (kind, evidence, frontier, input_order) that the engine
// consumes at impl/runtime/request_plan_impl.h:374-379 and the model never sees --
// it carries no token, no embedding and no position. SharedCandidateEvidence
// (include/ninfer/types.h:580-587) uses bits 0..4, so bit 5 is the first free one.
//
// This is NOT a new enum value: PromptCacheMarkerKind (include/ninfer/types.h:575-
// 578) is a closed two-value switch and adding a third would change the meaning of
// every existing marker, while an unused evidence BIT is additive and cannot
// collide. The static_assert below is the guard: if anyone claims bit 5, this stops
// compiling instead of silently aliasing a recall marker onto an existing reason.
inline constexpr std::uint8_t kRecallEvidenceBit = 1U << 5U;
static_assert((kRecallEvidenceBit & 0x1FU) == 0U,
              "SharedCandidateEvidence uses bits 0..4; the recall marker must take a free bit");
static_assert((kRecallEvidenceBit & ~0xFFU) == 0U,
              "SharedCandidateEvidence is a uint8_t; the recall marker must fit it");

// Token classes that must never carry a recall marker. Verified, not assumed:
// PreparedPromptData::token_types is uint8 per token; 0 means ordinary text and
// NON-ZERO means a Vision modality, because the prefill path looks up the matching
// Vision chunk and throws when it is missing (impl/runtime/text_prefill_impl.h:
// 349-363), and the reuse path tests `type != 0` to decide whether a vision
// re-projection is owed (impl/runtime/program_impl.h:4443-4447). So token_types is
// NOT a free annotation channel and a recall marker must not be written there.
inline constexpr std::uint8_t kOrdinaryTextTokenType = 0;

} // namespace ninfer::product
